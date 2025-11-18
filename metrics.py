"""
Metrics and evaluation utilities for comparing algorithms.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score, f1_score
from sklearn.cluster import KMeans
import time
import psutil
import os
import tracemalloc
from scipy.ndimage import binary_dilation

def compute_eigenvalue_accuracy(eigenvalues_computed, eigenvalues_true, k=None):
    """
    Compute accuracy of computed eigenvalues compared to ground truth.

    Parameters
    ----------
    eigenvalues_computed : ndarray
        Computed eigenvalues
    eigenvalues_true : ndarray
        True eigenvalues (from reference method)
    k : int, optional
        Number of eigenvalues to compare. If None, uses min of both lengths.

    Returns
    -------
    max_error : float
        Maximum absolute error
    mean_error : float
        Mean absolute error
    relative_error : float
        Mean relative error
    """
    if k is None:
        k = min(len(eigenvalues_computed), len(eigenvalues_true))

    evals_comp = eigenvalues_computed[:k]
    evals_true = eigenvalues_true[:k]

    abs_errors = np.abs(evals_comp - evals_true)
    max_error = np.max(abs_errors)
    mean_error = np.mean(abs_errors)

    # Relative error (avoid division by zero)
    mask = evals_true != 0
    if np.any(mask):
        relative_error = np.mean(
            np.abs(evals_comp[mask] - evals_true[mask]) / np.abs(evals_true[mask])
        )
    else:
        relative_error = mean_error

    return max_error, mean_error, relative_error


def compute_eigenvector_accuracy(eigenvectors_computed, eigenvectors_true, k=None):
    """
    Compute accuracy of computed eigenvectors.

    Uses subspace angle and correlation to measure accuracy.

    Parameters
    ----------
    eigenvectors_computed : ndarray
        Computed eigenvectors (n x k)
    eigenvectors_true : ndarray
        True eigenvectors (n x k)
    k : int, optional
        Number of eigenvectors to compare

    Returns
    -------
    max_angle : float
        Maximum principal angle between subspaces (in radians)
    mean_correlation : float
        Mean absolute correlation between corresponding eigenvectors
    """
    if k is None:
        k = min(eigenvectors_computed.shape[1], eigenvectors_true.shape[1])

    V_comp = eigenvectors_computed[:, :k]
    V_true = eigenvectors_true[:, :k]

    # Normalize
    V_comp = V_comp / np.linalg.norm(V_comp, axis=0, keepdims=True)
    V_true = V_true / np.linalg.norm(V_true, axis=0, keepdims=True)

    # Compute correlations
    correlations = np.abs(np.diag(V_comp.T @ V_true))
    mean_correlation = np.mean(correlations)

    # Compute principal angles between subspaces
    # Using SVD of V_comp^T V_true
    U, s, Vt = np.linalg.svd(V_comp.T @ V_true)
    angles = np.arccos(np.clip(s, -1, 1))
    max_angle = np.max(angles)

    return max_angle, mean_correlation


def spectral_clustering(eigenvectors, n_clusters, random_state=42):
    """
    Perform spectral clustering using k-means on eigenvectors.

    Parameters
    ----------
    eigenvectors : ndarray
        Eigenvectors (n x k) where k >= n_clusters
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed for k-means

    Returns
    -------
    labels : ndarray
        Cluster labels
    """
    # Use first n_clusters eigenvectors
    features = eigenvectors[:, :n_clusters]

    # Normalize rows
    row_norms = np.linalg.norm(features, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1  # Avoid division by zero
    features_normalized = features / row_norms

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_normalized)

    return labels

def labels_to_pixel_map(sp, sp_cluster_labels):
    H, W = sp.shape
    assert sp_cluster_labels.ndim == 1
    out = sp_cluster_labels[sp]
    return out.astype(np.int32)

def label_boundaries(label_img):
    """Binary boundary map from pixel labels (4-neighborhood)."""
    H, W = label_img.shape
    B = np.zeros((H, W), dtype=bool)
    B[:, 1:] |= (label_img[:, 1:] != label_img[:, :-1])
    B[1:, :] |= (label_img[1:, :] != label_img[:-1, :])
    return B

def boundary_f1(pred_labels, gt_labels, tol=1):
    Bp = label_boundaries(pred_labels)
    Bg = label_boundaries(gt_labels)
    if tol > 0:
        Bg = binary_dilation(Bg, iterations=tol)
        Bp = binary_dilation(Bp, iterations=tol)
    y_pred = Bp.ravel().astype(np.uint8)
    y_true = Bg.ravel().astype(np.uint8)
    return f1_score(y_true, y_pred)

def compute_segmentation_metrics(labels_pred, labels_true=None, image_shape=None):
    """
    Compute metrics for image segmentation quality.

    Parameters
    ----------
    labels_pred : ndarray
        Predicted cluster labels
    labels_true : ndarray, optional
        Ground truth labels (if available)
    image_shape : tuple, optional
        Shape of original image for visualization

    Returns
    -------
    metrics : dict
        Dictionary of metrics
    """
    metrics = {}

    # Number of clusters
    n_clusters = len(np.unique(labels_pred))
    metrics["n_clusters"] = n_clusters

    # Cluster sizes
    unique, counts = np.unique(labels_pred, return_counts=True)
    metrics["cluster_sizes"] = dict(zip(unique, counts))
    metrics["cluster_size_std"] = np.std(counts)

    # Adjusted Rand Index (if ground truth available)
    if labels_true is not None:
        metrics["adjusted_rand_score"] = adjusted_rand_score(labels_true, labels_pred)

    return metrics


def measure_runtime(func, *args, **kwargs):
    """
    Measure the runtime of a function.

    Parameters
    ----------
    func : callable
        Function to measure
    *args, **kwargs
        Arguments to pass to function

    Returns
    -------
    result : any
        Return value of function
    runtime : float
        Runtime in seconds
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    runtime = end_time - start_time
    return result, runtime

def measure_peak_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def compute_orthogonalization_loss(eigenvectors):
    """
    Measures deviation from perfect orthonormality.
    
    Returns Frobenius norm of (V^T V - I).
    Lower = better orthogonality.
    """
    V = eigenvectors
    k = V.shape[1]

    G = V.T @ V  # Gram matrix
    I = np.eye(k)

    loss = np.linalg.norm(G - I, ord='fro')

    return loss


def compare_algorithms(
    laplacian, k, algorithms, reference_eigenvals=None, reference_eigenvecs=None
):
    """
    Compare multiple eigenvalue algorithms.

    Parameters
    ----------
    laplacian : ndarray or sparse matrix
        Graph Laplacian
    k : int
        Number of eigenvectors to compute
    algorithms : dict
        Dictionary mapping algorithm names to functions
        Each function should take (laplacian, k) and return (eigenvals, eigenvecs, n_iter, history)
    reference_eigenvals : ndarray, optional
        Reference eigenvalues for accuracy comparison
    reference_eigenvecs : ndarray, optional
        Reference eigenvectors for accuracy comparison

    Returns
    -------
    results : dict
        Dictionary of results for each algorithm
    """
    results = {}

    for name, algorithm_func in algorithms.items():
        print(f"Running {name}...")
        max_iter_budget = getattr(algorithm_func, "max_iter_budget", None)

        # Measure runtime & memory
        start_mem = measure_peak_memory()
        tracemalloc.start()
        start_time = time.perf_counter()
        eigenvals, eigenvecs, n_iter, history = algorithm_func(laplacian)
        end_mem = measure_peak_memory()
        current_python, peak_python = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        runtime = time.perf_counter() - start_time
        peak_mem = max(start_mem, end_mem)

        # measure orthogonality loss
        ortho_loss = compute_orthogonalization_loss(eigenvecs)

        result = {
            "eigenvalues": eigenvals,
            "eigenvectors": eigenvecs,
            "n_iterations": n_iter,
            "runtime": runtime,
            "convergence_history": history,
            "peak_memory": peak_mem,
            "peak_python_memory": peak_python,
            "orthogonalization_loss": ortho_loss,
        }
        if max_iter_budget is not None:
            result["max_iter_budget"] = max_iter_budget
            result["converged"] = n_iter < max_iter_budget

        # Compute accuracy if reference available
        if reference_eigenvals is not None:
            max_err, mean_err, rel_err = compute_eigenvalue_accuracy(
                eigenvals, reference_eigenvals, k=k
            )
            result["eigenvalue_max_error"] = max_err
            result["eigenvalue_mean_error"] = mean_err
            result["eigenvalue_relative_error"] = rel_err

        if reference_eigenvecs is not None:
            max_angle, mean_corr = compute_eigenvector_accuracy(
                eigenvecs, reference_eigenvecs, k=k
            )
            result["eigenvector_max_angle"] = max_angle
            result["eigenvector_mean_correlation"] = mean_corr

        results[name] = result

    return results

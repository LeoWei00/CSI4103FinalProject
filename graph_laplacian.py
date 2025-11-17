"""
Graph Laplacian construction for image segmentation using k-nearest neighbors.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from skimage import img_as_float, segmentation, color


def image_to_feature_vectors(image, return_superpixel_labels=False):
    """
    Convert an image to feature vectors for graph construction.

    Parameters
    ----------
    image : ndarray
        Image array of shape (height, width, channels) or (height, width)

    Returns
    -------
    features : ndarray
        Feature vectors of shape (n_superpixels, n_features)
    labels : ndarray, optional
        SLIC superpixel labels per pixel if return_superpixel_labels is True
    """
    img_f = img_as_float(image)

    if img_f.ndim == 2:
        channel_axis = None
        feature_image = img_f[..., None]
    elif img_f.ndim == 3:
        channel_axis = -1
        if img_f.shape[-1] == 3:
            feature_image = color.rgb2lab(img_f)
        else:
            feature_image = img_f
    else:
        raise ValueError("Unsupported image shape for feature extraction.")

    labels = segmentation.slic(
        img_f,
        n_segments=800,
        compactness=10.0,
        sigma=1.0,
        channel_axis=channel_axis,
        start_label=0,
    )

    H, W = labels.shape
    N = int(labels.max()) + 1

    yy, xx = np.indices((H, W))
    labels_flat = labels.ravel()
    counts = np.bincount(labels_flat, minlength=N)

    feature_pixels = feature_image.reshape(-1, feature_image.shape[-1])
    sums = np.zeros((N, feature_pixels.shape[1]), dtype=np.float64)
    for c in range(feature_pixels.shape[1]):
        sums[:, c] = np.bincount(
            labels_flat,
            weights=feature_pixels[:, c],
            minlength=N,
        )
    mean_features = sums / np.maximum(counts[:, None], 1)

    cx = np.bincount(labels_flat, weights=xx.ravel(), minlength=N) / np.maximum(counts, 1)
    cy = np.bincount(labels_flat, weights=yy.ravel(), minlength=N) / np.maximum(counts, 1)
    centers_xy = np.stack([cx, cy], axis=1)

    xy = centers_xy.astype(np.float64)
    xy[:, 0] /= max(W, 1)
    xy[:, 1] /= max(H, 1)

    features = np.concatenate([mean_features, xy], axis=1)

    if return_superpixel_labels:
        return features, labels
    return features


def construct_knn_graph(features, k=10, sigma=None):
    """
    Construct a k-nearest neighbor graph from feature vectors.

    Parameters
    ----------
    features : ndarray
        Feature vectors of shape (n_samples, n_features)
    k : int
        Number of nearest neighbors
    sigma : float, optional
        Bandwidth parameter for Gaussian weights. If None, uses median distance.

    Returns
    -------
    adjacency : csr_matrix
        Sparse adjacency matrix
    """
    n_samples = features.shape[0]

    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)

    # Remove self-connections (first neighbor is always the point itself)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Set sigma to median distance if not provided
    if sigma is None:
        sigma = np.median(distances)

    # Compute Gaussian weights
    weights = np.exp(-(distances**2) / (2 * sigma**2))

    # Build sparse adjacency matrix
    row_indices = np.repeat(np.arange(n_samples), k)
    col_indices = indices.flatten()
    data = weights.flatten()

    # Make symmetric (undirected graph)
    adjacency = csr_matrix(
        (data, (row_indices, col_indices)), shape=(n_samples, n_samples)
    )
    adjacency = (adjacency + adjacency.T) / 2  # Average to ensure symmetry

    return adjacency


def construct_laplacian(adjacency, normalized=True):
    """
    Construct graph Laplacian from adjacency matrix.

    Parameters
    ----------
    adjacency : csr_matrix or ndarray
        Adjacency matrix
    normalized : bool
        If True, compute normalized Laplacian L = I - D^(-1/2) A D^(-1/2)
        If False, compute unnormalized Laplacian L = D - A

    Returns
    -------
    laplacian : csr_matrix
        Graph Laplacian matrix
    """
    if not isinstance(adjacency, csr_matrix):
        adjacency = csr_matrix(adjacency)

    # Compute degree matrix
    degrees = np.array(adjacency.sum(axis=1)).flatten()

    if normalized:
        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        # Avoid division by zero
        degrees_sqrt_inv = np.zeros_like(degrees)
        mask = degrees > 0
        degrees_sqrt_inv[mask] = 1.0 / np.sqrt(degrees[mask])

        # D^(-1/2) A D^(-1/2)
        D_inv_sqrt = csr_matrix(
            (degrees_sqrt_inv, (np.arange(len(degrees)), np.arange(len(degrees))))
        )
        laplacian = D_inv_sqrt @ adjacency @ D_inv_sqrt

        # L = I - D^(-1/2) A D^(-1/2)
        n = adjacency.shape[0]
        identity = csr_matrix(np.eye(n))
        laplacian = identity - laplacian
    else:
        # Unnormalized Laplacian: L = D - A
        D = csr_matrix((degrees, (np.arange(len(degrees)), np.arange(len(degrees)))))
        laplacian = D - adjacency

    return laplacian


def image_to_laplacian(
    image, k=10, sigma=None, normalized=True, return_superpixel_labels=False
):
    """
    Convert an image directly to a graph Laplacian.

    Parameters
    ----------
    image : ndarray
        Image array
    k : int
        Number of nearest neighbors
    sigma : float, optional
        Bandwidth parameter for Gaussian weights
    normalized : bool
        Whether to use normalized Laplacian

    Returns
    -------
    laplacian : csr_matrix
        Graph Laplacian matrix
    labels : ndarray, optional
        Superpixel labels per pixel if return_superpixel_labels is True
    """
    if return_superpixel_labels:
        features, slic_labels = image_to_feature_vectors(
            image, return_superpixel_labels=True
        )
    else:
        features = image_to_feature_vectors(image)
    adjacency = construct_knn_graph(features, k=k, sigma=sigma)
    laplacian = construct_laplacian(adjacency, normalized=normalized)
    if return_superpixel_labels:
        return laplacian, slic_labels
    return laplacian

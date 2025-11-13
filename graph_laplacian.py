"""
Graph Laplacian construction for image segmentation using k-nearest neighbors.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def image_to_feature_vectors(image):
    """
    Convert an image to feature vectors for graph construction.

    Parameters
    ----------
    image : ndarray
        Image array of shape (height, width, channels) or (height, width)

    Returns
    -------
    features : ndarray
        Feature vectors of shape (n_pixels, n_features)
        Features include spatial coordinates and pixel intensities
    """
    if len(image.shape) == 2:
        # Grayscale image
        height, width = image.shape
        channels = 1
        image = image.reshape(height, width, 1)
    else:
        height, width, channels = image.shape

    # Create coordinate grid
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Normalize coordinates to [0, 1]
    y_coords = y_coords.astype(float) / height
    x_coords = x_coords.astype(float) / width

    # Normalize pixel intensities to [0, 1]
    image_normalized = image.astype(float) / 255.0

    # Combine spatial and intensity features
    features = np.column_stack(
        [y_coords.flatten(), x_coords.flatten(), image_normalized.reshape(-1, channels)]
    )

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


def image_to_laplacian(image, k=10, sigma=None, normalized=True):
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
    """
    features = image_to_feature_vectors(image)
    adjacency = construct_knn_graph(features, k=k, sigma=sigma)
    laplacian = construct_laplacian(adjacency, normalized=normalized)
    return laplacian

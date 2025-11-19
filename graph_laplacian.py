"""
Graph Laplacian construction for image segmentation using k-nearest neighbors.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from skimage import img_as_float, segmentation, color
from scipy.sparse import csr_matrix, eye as sparse_eye


def image_to_feature_vectors(
    image,
    return_superpixel_labels=False,
    n_segments=800,
    use_superpixels=True
):
    """
    Convert an image to feature vectors for graph construction.

    Parameters
    ----------
    image : ndarray
        Image array (H, W) or (H, W, C)
    return_superpixel_labels : bool
        If True, return labels array
    n_segments : int
        Number of superpixels (only used if use_superpixels=True)
    use_superpixels : bool
        If False, each pixel becomes one feature vector (no SLIC)

    Returns
    -------
    features : ndarray (N, d)
    labels   : ndarray (H, W) if requested
    """
    img_f = img_as_float(image)

    # Convert to Lab if RGB
    if img_f.ndim == 2:
        channel_axis = None
        feature_image = img_f[..., None]    # shape (H, W, 1)
    elif img_f.ndim == 3:
        channel_axis = -1
        if img_f.shape[-1] == 3:
            feature_image = color.rgb2lab(img_f)
        else:
            feature_image = img_f
    else:
        raise ValueError("Unsupported image shape for feature extraction.")

    H, W = feature_image.shape[:2]

    # -------------------------------------------------------
    # OPTION 1: NO SUPERPIXELS (use each pixel individually)
    # -------------------------------------------------------
    if not use_superpixels:

        # Create pixel-level labels (0 .. H*W-1)
        labels = np.arange(H * W).reshape(H, W)

        # Features: pixel value(s) + xy-coordinates
        feature_pixels = feature_image.reshape(-1, feature_image.shape[-1])

        yy, xx = np.indices((H, W))
        xy = np.stack([xx.ravel() / W, yy.ravel() / H], axis=1)

        features = np.concatenate([feature_pixels, xy], axis=1)

        if return_superpixel_labels:
            return features, labels
        return features

    # -------------------------------------------------------
    # OPTION 2: USE SUPERPIXELS (SLIC)
    # -------------------------------------------------------

    labels = segmentation.slic(
        img_f,
        n_segments=n_segments,
        compactness=10.0,
        sigma=1.0,
        channel_axis=channel_axis,
        start_label=0,
    )

    N = int(labels.max()) + 1
    labels_flat = labels.ravel()

    # Pixel features
    feature_pixels = feature_image.reshape(-1, feature_image.shape[-1])

    # Count pixels per superpixel
    counts = np.bincount(labels_flat, minlength=N)

    # Average pixel features per superpixel
    sums = np.zeros((N, feature_pixels.shape[1]), dtype=np.float64)
    for c in range(feature_pixels.shape[1]):
        sums[:, c] = np.bincount(
            labels_flat,
            weights=feature_pixels[:, c],
            minlength=N,
        )

    mean_features = sums / np.maximum(counts[:, None], 1)

    # Superpixel centers (x,y)
    yy, xx = np.indices((H, W))
    cx = np.bincount(labels_flat, weights=xx.ravel(), minlength=N) / np.maximum(counts, 1)
    cy = np.bincount(labels_flat, weights=yy.ravel(), minlength=N) / np.maximum(counts, 1)
    xy = np.stack([cx / W, cy / H], axis=1)

    features = np.concatenate([mean_features, xy], axis=1)

    if return_superpixel_labels:
        return features, labels

    return features


def construct_knn_graph(features, k=10, sigma=None):
    n_samples = features.shape[0]

    # k cannot exceed n_samples - 1
    if n_samples <= 1:
        raise ValueError(f"Not enough samples to build a graph: n_samples={n_samples}")
    k_eff = min(k, n_samples - 1)

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto", metric="euclidean")
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)

    # drop self neighbor
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    if sigma is None:
        sigma = np.median(distances)

    weights = np.exp(-(distances**2) / (2 * sigma**2))

    row_indices = np.repeat(np.arange(n_samples), k_eff)
    col_indices = indices.flatten()
    data = weights.flatten()

    adjacency = csr_matrix(
        (data, (row_indices, col_indices)), shape=(n_samples, n_samples)
    )
    adjacency = (adjacency + adjacency.T) / 2

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
    image,
    k=10,
    sigma=None,
    normalized=True,
    return_superpixel_labels=False,
    n_segments=800,
    use_superpixels=True
):
    if return_superpixel_labels:
        features, slic_labels = image_to_feature_vectors(
            image,
            return_superpixel_labels=True,
            n_segments=n_segments,
            use_superpixels=use_superpixels
        )
    else:
        features = image_to_feature_vectors(
            image,
            use_superpixels=use_superpixels,
            n_segments=n_segments
        )

    adjacency = construct_knn_graph(features, k=k, sigma=sigma)
    laplacian = construct_laplacian(adjacency, normalized=normalized)

    if return_superpixel_labels:
        return laplacian, slic_labels
    return laplacian
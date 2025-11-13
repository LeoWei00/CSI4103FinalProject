"""
Subspace Iteration algorithms for computing eigenvectors.
"""

import numpy as np
from scipy.linalg import qr, solve
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import spsolve


def standard_subspace_iteration(A, k, max_iter=1000, tol=1e-10, initial_vectors=None):
    """
    Standard subspace iteration (power method for multiple vectors).

    Computes the k smallest eigenvalues and corresponding eigenvectors of A.
    For symmetric matrices, this finds the k smallest eigenvalues.

    Parameters
    ----------
    A : ndarray or sparse matrix
        Symmetric matrix (typically graph Laplacian)
    k : int
        Number of eigenvectors to compute
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    initial_vectors : ndarray, optional
        Initial guess for eigenvectors (n x k)

    Returns
    -------
    eigenvalues : ndarray
        k smallest eigenvalues
    eigenvectors : ndarray
        Corresponding eigenvectors (n x k)
    n_iter : int
        Number of iterations performed
    convergence_history : list
        History of eigenvalue estimates
    """
    if issparse(A):
        A = A.toarray()

    n = A.shape[0]

    # Initialize random vectors if not provided
    if initial_vectors is None:
        Q = np.random.randn(n, k)
    else:
        Q = initial_vectors.copy()

    # Orthonormalize initial vectors
    Q, _ = qr(Q, mode="economic")

    convergence_history = []
    eigenvalues_old = np.zeros(k)

    for iteration in range(max_iter):
        # Multiply by A
        AQ = A @ Q

        # Orthonormalize
        Q, R = qr(AQ, mode="economic")

        # Estimate eigenvalues using Rayleigh quotients
        eigenvalues = np.array([Q[:, i].T @ A @ Q[:, i] for i in range(k)])

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        Q = Q[:, idx]

        convergence_history.append(eigenvalues.copy())

        # Check convergence
        if iteration > 0:
            error = np.max(np.abs(eigenvalues - eigenvalues_old))
            if error < tol:
                break

        eigenvalues_old = eigenvalues.copy()

    return eigenvalues, Q, iteration + 1, convergence_history


def block_subspace_iteration(
    A, k, max_iter=1000, tol=1e-10, initial_vectors=None, block_size=None
):
    """
    Block subspace iteration for computing multiple eigenvectors.

    This is more efficient than standard subspace iteration when computing
    multiple eigenvectors, as it processes them in blocks.

    Parameters
    ----------
    A : ndarray or sparse matrix
        Symmetric matrix (typically graph Laplacian)
    k : int
        Number of eigenvectors to compute
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    initial_vectors : ndarray, optional
        Initial guess for eigenvectors (n x k)
    block_size : int, optional
        Size of blocks for processing. If None, uses k.

    Returns
    -------
    eigenvalues : ndarray
        k smallest eigenvalues
    eigenvectors : ndarray
        Corresponding eigenvectors (n x k)
    n_iter : int
        Number of iterations performed
    convergence_history : list
        History of eigenvalue estimates
    """
    if issparse(A):
        A = A.toarray()

    n = A.shape[0]

    if block_size is None:
        block_size = k

    # Initialize random vectors if not provided
    if initial_vectors is None:
        Q = np.random.randn(n, k)
    else:
        Q = initial_vectors.copy()

    # Orthonormalize initial vectors
    Q, _ = qr(Q, mode="economic")

    convergence_history = []
    eigenvalues_old = np.zeros(k)

    for iteration in range(max_iter):
        # Block processing: multiply by A
        AQ = A @ Q

        # Orthonormalize using QR decomposition
        Q, R = qr(AQ, mode="economic")

        # Project A onto the subspace spanned by Q
        # A_proj = Q^T A Q
        A_proj = Q.T @ A @ Q

        # Solve eigenvalue problem in the projected subspace
        # This is much smaller (k x k) than the original problem
        eigenvals_proj, eigenvecs_proj = np.linalg.eigh(A_proj)

        # Sort by eigenvalue
        idx = np.argsort(eigenvals_proj)
        eigenvals_proj = eigenvals_proj[idx]
        eigenvecs_proj = eigenvecs_proj[:, idx]

        # Update Q to be the eigenvectors in the original space
        Q = Q @ eigenvecs_proj

        # The projected eigenvalues are our estimates
        eigenvalues = eigenvals_proj[:k]

        convergence_history.append(eigenvalues.copy())

        # Check convergence
        if iteration > 0:
            error = np.max(np.abs(eigenvalues - eigenvalues_old))
            if error < tol:
                break

        eigenvalues_old = eigenvalues.copy()

    return eigenvalues, Q, iteration + 1, convergence_history

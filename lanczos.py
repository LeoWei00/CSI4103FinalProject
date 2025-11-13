"""
Lanczos method for computing eigenvalues and eigenvectors of symmetric matrices.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import LinearOperator


def lanczos_iteration(A, k, max_iter=None, tol=1e-10, initial_vector=None):
    """
    Lanczos method for computing k smallest eigenvalues and eigenvectors.

    The Lanczos method builds a tridiagonal matrix whose eigenvalues
    approximate those of A. This is particularly efficient for sparse matrices.

    Parameters
    ----------
    A : ndarray or sparse matrix
        Symmetric matrix
    k : int
        Number of eigenvalues/eigenvectors to compute
    max_iter : int, optional
        Maximum number of Lanczos iterations. If None, uses min(n, 2*k+10)
    tol : float
        Convergence tolerance
    initial_vector : ndarray, optional
        Initial starting vector

    Returns
    -------
    eigenvalues : ndarray
        k smallest eigenvalues
    eigenvectors : ndarray
        Corresponding eigenvectors (n x k)
    n_iter : int
        Number of Lanczos iterations performed
    convergence_history : list
        History of eigenvalue estimates
    """
    if issparse(A):
        n = A.shape[0]
        matvec = lambda x: A @ x
    else:
        n = A.shape[0]
        matvec = lambda x: A @ x

    if max_iter is None:
        max_iter = min(n, 2 * k + 10)
    max_iter = min(max_iter, n)

    # Initialize
    if initial_vector is None:
        v = np.random.randn(n)
    else:
        v = initial_vector.copy()

    v = v / np.linalg.norm(v)

    # Storage for Lanczos vectors
    V = np.zeros((n, max_iter))
    V[:, 0] = v

    # Tridiagonal matrix storage
    alpha = np.zeros(max_iter)
    beta = np.zeros(max_iter - 1)

    convergence_history = []

    # First iteration
    w = matvec(v)
    alpha[0] = v.T @ w
    w = w - alpha[0] * v

    for j in range(1, max_iter):
        beta[j - 1] = np.linalg.norm(w)

        # Check for breakdown
        if beta[j - 1] < tol:
            # Restart with random vector
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)
            V[:, j] = v
            w = matvec(v)
            alpha[j] = v.T @ w
            w = w - alpha[j] * v
            continue

        v = w / beta[j - 1]
        V[:, j] = v

        w = matvec(v)
        alpha[j] = v.T @ w
        w = w - alpha[j] * v - beta[j - 1] * V[:, j - 1]

        # Build tridiagonal matrix
        T = np.zeros((j + 1, j + 1))
        for i in range(j + 1):
            T[i, i] = alpha[i]
            if i < j:
                T[i, i + 1] = beta[i]
                T[i + 1, i] = beta[i]

        # Compute eigenvalues of tridiagonal matrix
        eigenvals_T, _ = eigh(T)
        eigenvals_T = np.sort(eigenvals_T)

        # Track convergence of k smallest eigenvalues
        k_smallest = eigenvals_T[: min(k, len(eigenvals_T))]
        convergence_history.append(k_smallest.copy())

        # Check convergence (if we have enough eigenvalues)
        if len(eigenvals_T) >= k and j > k:
            # Check if eigenvalues have converged
            if len(convergence_history) > 1:
                prev_vals = convergence_history[-2]
                if len(prev_vals) >= k:
                    error = np.max(np.abs(k_smallest[:k] - prev_vals[:k]))
                    if error < tol:
                        break

    # Final tridiagonal matrix
    m = j + 1
    T = np.zeros((m, m))
    for i in range(m):
        T[i, i] = alpha[i]
        if i < m - 1:
            T[i, i + 1] = beta[i]
            T[i + 1, i] = beta[i]

    # Compute eigenvalues and eigenvectors of tridiagonal matrix
    eigenvals_T, eigenvecs_T = eigh(T)

    # Sort by eigenvalue
    idx = np.argsort(eigenvals_T)
    eigenvals_T = eigenvals_T[idx]
    eigenvecs_T = eigenvecs_T[:, idx]

    # Select k smallest
    eigenvalues = eigenvals_T[:k]
    eigenvecs_T = eigenvecs_T[:, :k]

    # Transform back to original space
    eigenvectors = V[:, :m] @ eigenvecs_T

    # Orthonormalize (Lanczos can lose orthogonality)
    eigenvectors, _ = np.linalg.qr(eigenvectors)

    return eigenvalues, eigenvectors, j + 1, convergence_history

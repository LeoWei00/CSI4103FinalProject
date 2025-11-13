"""
Custom QR iteration algorithm for computing eigenvalues and eigenvectors.
"""

import numpy as np
from scipy.linalg import qr, hessenberg
from scipy.sparse import issparse


def qr_iteration(A, max_iter=1000, tol=1e-10, shift=True):
    """
    QR iteration algorithm for computing all eigenvalues and eigenvectors.

    Parameters
    ----------
    A : ndarray
        Square matrix
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    shift : bool
        Whether to use Wilkinson shift for faster convergence

    Returns
    -------
    eigenvalues : ndarray
        All eigenvalues
    eigenvectors : ndarray
        Corresponding eigenvectors (columns)
    n_iter : int
        Number of iterations performed
    """
    if issparse(A):
        A = A.toarray()

    A = A.copy()
    n = A.shape[0]

    # Reduce to Hessenberg form for efficiency
    H, Q_hess = hessenberg(A, calc_q=True)

    # Track eigenvectors
    Q_total = Q_hess.copy()

    for iteration in range(max_iter):
        # Check for convergence (check if lower triangular part is small)
        off_diag_norm = np.linalg.norm(np.tril(H, -1))
        if off_diag_norm < tol:
            break

        # Wilkinson shift for faster convergence
        if shift and n > 1:
            # Use bottom 2x2 submatrix for shift
            a = H[n - 2, n - 2]
            b = H[n - 2, n - 1]
            c = H[n - 1, n - 2]
            d = H[n - 1, n - 1]

            # Eigenvalues of 2x2 matrix
            trace = a + d
            det = a * d - b * c
            discriminant = trace**2 - 4 * det

            if discriminant >= 0:
                # Real eigenvalues
                lambda1 = (trace + np.sqrt(discriminant)) / 2
                lambda2 = (trace - np.sqrt(discriminant)) / 2
                # Choose shift closer to d
                shift_val = lambda2 if abs(lambda2 - d) < abs(lambda1 - d) else lambda1
            else:
                # Complex eigenvalues, use d as shift
                shift_val = d
        else:
            shift_val = 0.0

        # Apply shift
        H_shifted = H - shift_val * np.eye(n)

        # QR decomposition
        Q, R = qr(H_shifted)

        # Reverse QR: RQ + shift
        H = R @ Q + shift_val * np.eye(n)

        # Update eigenvectors
        Q_total = Q_total @ Q

    # Extract eigenvalues from diagonal
    eigenvalues = np.diag(H)

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = Q_total[:, idx]

    return eigenvalues, eigenvectors, iteration + 1


def qr_iteration_partial(A, k, max_iter=1000, tol=1e-10):
    """
    QR iteration to compute only the k smallest eigenvalues and eigenvectors.

    This is less efficient than full QR, but included for comparison.
    In practice, you would use deflation or other techniques.

    Parameters
    ----------
    A : ndarray
        Square symmetric matrix
    k : int
        Number of smallest eigenvalues to compute
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    eigenvalues : ndarray
        k smallest eigenvalues
    eigenvectors : ndarray
        Corresponding eigenvectors (columns)
    n_iter : int
        Number of iterations performed
    """
    # Compute all eigenvalues/eigenvectors, then select k smallest
    eigenvalues, eigenvectors, n_iter = qr_iteration(A, max_iter=max_iter, tol=tol)

    # Sort and select k smallest
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx][:k]
    eigenvectors = eigenvectors[:, idx][:, :k]

    return eigenvalues, eigenvectors, n_iter

"""
practical_qr.py

Lanczos tridiagonalization + implicit QR on the small tridiagonal T.
"""

import numpy as np
from scipy.sparse import issparse
from scipy.linalg import qr

# ---------------------------------------------------------
# 1. Lanczos tridiagonalization (no restarting)
# ---------------------------------------------------------

def lanczos_tridiagonal(A, m, v0=None, tol=1e-14):
    """
    Run m steps of (plain) Lanczos on symmetric A.

    Parameters
    ----------
    A : (n, n) ndarray or sparse matrix
        Symmetric matrix (e.g. Laplacian).
    m : int
        Dimension of Lanczos subspace (number of steps).
    v0 : ndarray, optional
        Initial vector (length n). If None, random.
    tol : float
        Breakdown tolerance.

    Returns
    -------
    V : ndarray, shape (n, m)
        Orthonormal Lanczos basis vectors.
    alpha : ndarray, shape (m,)
        Diagonal entries of the tridiagonal matrix T.
    beta : ndarray, shape (m-1,)
        Off-diagonal entries of T.
    """
    if issparse(A):
        matvec = lambda x: A @ x
        n = A.shape[0]
    else:
        A = np.asarray(A)
        matvec = lambda x: A @ x
        n = A.shape[0]

    if v0 is None:
        v = np.random.randn(n)
    else:
        v = np.array(v0, dtype=float)

    v /= np.linalg.norm(v)

    V = np.zeros((n, m))
    alpha = np.zeros(m)
    beta = np.zeros(m - 1)

    V[:, 0] = v
    w = matvec(v)
    alpha[0] = v @ w
    w = w - alpha[0] * v

    for j in range(1, m):
        beta[j - 1] = np.linalg.norm(w)
        if beta[j - 1] < tol:
            # simple breakdown handling: restart with random vector
            v = np.random.randn(n)
            v /= np.linalg.norm(v)
            V[:, j] = v
            w = matvec(v)
            alpha[j] = v @ w
            w = w - alpha[j] * v
            continue

        v = w / beta[j - 1]
        V[:, j] = v
        w = matvec(v)
        alpha[j] = v @ w
        w = w - alpha[j] * v - beta[j - 1] * V[:, j - 1]

    return V, alpha, beta


def build_tridiagonal(alpha, beta):
    """Construct the symmetric tridiagonal matrix T from alpha, beta."""
    m = len(alpha)
    T = np.zeros((m, m))
    for i in range(m):
        T[i, i] = alpha[i]
        if i < m - 1:
            T[i, i + 1] = beta[i]
            T[i + 1, i] = beta[i]
    return T


# ---------------------------------------------------------
# 2. Implicit QR iteration on small symmetric tridiagonal T
# ---------------------------------------------------------

def qr_iteration_tridiagonal(T, max_iter=1000, tol=1e-12):
    """
    Implicitly shifted QR on a small symmetric matrix T.

    This is a *practical* implementation, not super optimized,
    but fine for the small m (~k+10) coming from Lanczos.

    Parameters
    ----------
    T : (m, m) ndarray
        Symmetric (tridiagonal) matrix.
    max_iter : int
        Maximum QR iterations.
    tol : float
        Convergence tolerance on off-diagonal norm.

    Returns
    -------
    evals : ndarray, shape (m,)
        Approximate eigenvalues (diagonal of reduced matrix).
    evecs : ndarray, shape (m, m)
        Approximate eigenvectors (columns).
    history : list of ndarray
        History of smallest few eigenvalues per iteration (for plotting).
    """
    A_k = T.copy()
    m = A_k.shape[0]
    Q_total = np.eye(m)
    history = []

    for it in range(max_iter):
        # simple Wilkinson-style shift: use bottom-right element
        mu = A_k[-1, -1]

        Q, R = np.linalg.qr(A_k - mu * np.eye(m))
        A_k = R @ Q + mu * np.eye(m)

        Q_total = Q_total @ Q

        # track (say) first 5 eigenvalues estimate, just for convergence plots
        diag_vals = np.diag(A_k)
        history.append(np.sort(diag_vals.copy()))

        # check off-diagonal norm for convergence
        off_norm = np.sqrt(np.sum(A_k**2) - np.sum(diag_vals**2))
        if off_norm < tol:
            break

    evals = np.diag(A_k)
    evecs = Q_total

    return evals, evecs, it + 1, history


# ---------------------------------------------------------
# 3. Practical QR path: Lanczos → T → implicit QR(T) → Ritz pairs
# ---------------------------------------------------------

def lanczos_practical_qr(
    A,
    k,
    m=None,
    max_lanczos=None,   # kept for symmetry with other solvers; not really used
    max_qr_iter=1000,
    tol=1e-10,
):
    """
    Practical QR path for symmetric A:

        Lanczos tridiagonalization  →  T
        implicit QR on T            →  eigenpairs of T
        Ritz pairs                  →  approximate eigenpairs of A

    This is your "QR Algorithm (practical interpretation)" solver.

    Parameters
    ----------
    A : (n, n) ndarray or sparse matrix
        Symmetric matrix (e.g., Laplacian).
    k : int
        Number of smallest eigenpairs to return.
    m : int, optional
        Lanczos subspace dimension (>= k). Default: k + 10 or n, whichever is smaller.
    max_lanczos : int, optional
        Included for API symmetry; currently unused (single Lanczos pass of size m).
    max_qr_iter : int
        Maximum QR iterations on T.
    tol : float
        Tolerance used inside QR iteration (off-diagonal norm).

    Returns
    -------
    eigenvalues : ndarray, shape (k,)
        Smallest k Rayleigh–Ritz eigenvalues approximating those of A.
    eigenvectors : ndarray, shape (n, k)
        Corresponding Ritz vectors in the original space.
    n_iter : int
        Number of QR iterations performed on T.
    convergence_history : list
        History of eigenvalue estimates from the QR iteration (on T).
    """
    if issparse(A):
        n = A.shape[0]
    else:
        A = np.asarray(A)
        n = A.shape[0]

    # Lanczos subspace size
    if m is None:
        m = min(n, k + 10)

    # 1) Lanczos tridiagonalization: A ≈ V T V^T
    V, alpha, beta = lanczos_tridiagonal(A, m)
    T = build_tridiagonal(alpha, beta)

    # 2) Implicit QR on the small T
    evals_T, evecs_T, n_qr_iter, qr_history = qr_iteration_tridiagonal(
        T, max_iter=max_qr_iter, tol=tol
    )

    # Sort eigenpairs of T (we want smallest eigenvalues of A, which
    # correspond to smallest eigenvalues of T for a good Lanczos run).
    idx = np.argsort(evals_T)
    evals_T = evals_T[idx]
    evecs_T = evecs_T[:, idx]

    # 3) Ritz vectors in original space: V * y_j
    ritz_vecs = V @ evecs_T    # shape (n, m)

    # take the first k (smallest eigenvalues)
    eigenvalues = evals_T[:k]
    eigenvectors = ritz_vecs[:, :k]

    # optionally orthonormalize Ritz vectors (helps numerically)
    eigenvectors, _ = qr(eigenvectors, mode="economic")

    return eigenvalues, eigenvectors, n_qr_iter, qr_history


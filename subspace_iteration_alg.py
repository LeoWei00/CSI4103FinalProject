"""
Subspace Iteration algorithms for computing eigenvectors.
"""

import numpy as np
from scipy.linalg import qr, solve
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import issparse, eye as sparse_eye
from scipy.linalg import qr, eigh
from scipy.sparse.linalg import eigsh


import numpy as np
from scipy.linalg import qr, eigh
from scipy.sparse import issparse, eye as sparse_eye
from scipy.sparse.linalg import eigsh

def standard_subspace_iteration(
    A,
    k,
    max_iter=1000,
    tol=1e-10,
    initial_vectors=None,
    skip_trivial=False,
):
    """
    Standard subspace iteration targeting the *smallest* k eigenvalues of a
    symmetric matrix A.

    For general symmetric A, it:
      - estimates the largest eigenvalue λ_max(A),
      - forms B = α I - A with α = λ_max,
      - runs subspace iteration on B,
      - performs Rayleigh–Ritz on A to extract the smallest eigenvalues.

    If A is a graph Laplacian (e.g. L or L_sym), you can set skip_trivial=True
    to drop the smallest (typically trivial) eigenvalue and return the next k.

    Parameters
    ----------
    A : ndarray or sparse matrix
        Symmetric matrix (e.g. graph Laplacian).
    k : int
        Number of eigenvalues/eigenvectors to return (nontrivial ones if
        skip_trivial=True).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on eigenvalues.
    initial_vectors : ndarray, optional
        Initial guess for the subspace (n x block_size).
    skip_trivial : bool
        If True, we assume the very smallest eigenvalue is trivial (e.g. 0 for
        Laplacians) and return the next k eigenpairs.

    Returns
    -------
    eigenvalues : ndarray, shape (k,)
        Smallest (or smallest nontrivial) eigenvalues of A.
    eigenvectors : ndarray, shape (n, k)
        Corresponding eigenvectors.
    n_iter : int
        Number of iterations performed.
    convergence_history : list of ndarray
        History of k-eigenvalue estimates per iteration.
    """
    # ---- Dense vs sparse setup ----
    if issparse(A):
        n = A.shape[0]
        I = sparse_eye(n, format="csr")
        A_mat = A
        # Estimate largest eigenvalue with ARPACK
        w_max, _ = eigsh(A_mat, k=1, which="LA")
        alpha = float(w_max[0])
    else:
        A_mat = np.asarray(A)
        n = A_mat.shape[0]
        I = np.eye(n)
        # Exact largest eigenvalue for small dense test matrices
        w_all = eigh(A_mat, eigvals_only=True)
        alpha = float(w_all[-1])  # largest

    # Subspace iteration on B = α I - A
    B = alpha * I - A_mat

    # Decide working subspace dimension
    # If we skip the trivial eigenvalue, we need room for it as well.
    block_size = k

    # ---- Initialize subspace Q ----
    if initial_vectors is None:
        Q = np.random.randn(n, block_size)
    else:
        Q = initial_vectors.copy()
        # If user-provided has wrong width, you may want to assert or adjust.

    Q, _ = qr(Q, mode="economic")

    convergence_history = []
    eigenvalues_old = None

    # ---- Subspace iteration loop ----
    for it in range(max_iter):
        # 1) Apply B to the subspace
        Q_new = B @ Q

        # 2) Orthonormalize
        Q, _ = qr(Q_new, mode="economic")

        # 3) Rayleigh–Ritz on original A:
        #    T = Q^T A Q is small (block_size x block_size)
        T = Q.T @ (A_mat @ Q)
        evals_proj, evecs_proj = eigh(T)

        # Sort ascending (smallest eigenvalues first)
        idx = np.argsort(evals_proj)
        evals_proj = evals_proj[idx]
        evecs_proj = evecs_proj[:, idx]

        # 4) Refine Q to approximate the eigenvectors of A
        Q = Q @ evecs_proj   # still n x block_size

        # 5) Extract the k eigenvalues we care about
        if skip_trivial:
            # assume the very smallest is trivial (e.g. 0)
            eigenvalues = evals_proj[1 : k + 1]
        else:
            eigenvalues = evals_proj[:k]

        convergence_history.append(eigenvalues.copy())

        # 6) Convergence check
        if eigenvalues_old is not None:
            err = np.max(np.abs(eigenvalues - eigenvalues_old))
            if err < tol:
                break

        eigenvalues_old = eigenvalues.copy()

    # ---- Final eigenvectors (matching those k eigenvalues) ----
    if skip_trivial:
        eigenvectors = Q[:, 1 : k + 1]
    else:
        eigenvectors = Q[:, :k]

    return eigenvalues, eigenvectors, it + 1, convergence_history



def block_subspace_iteration(
    A,
    k,
    max_iter=1000,
    tol=1e-10,
    block_size=None,
    skip_trivial=False,
):
    """
    Block subspace iteration for *smallest* k eigenvalues of symmetric A.

    If skip_trivial=True (typical for graph Laplacians), the routine
    computes k+1 Ritz values internally and returns the k smallest
    *nontrivial* ones (i.e. skips the very smallest).

    Parameters
    ----------
    A : ndarray or sparse matrix
        Symmetric matrix (e.g., Laplacian).
    k : int
        Number of eigenvalues/eigenvectors to return (nontrivial ones if
        skip_trivial=True).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on eigenvalues.
    block_size : int, optional
        Dimension of the working subspace. If None, uses
        k (+1 if skip_trivial).
    skip_trivial : bool
        If True, assume the smallest eigenvalue is trivial and return
        the next k eigenpairs.

    Returns
    -------
    eigenvalues : (k,) ndarray
        Smallest (or smallest nontrivial) eigenvalues.
    eigenvectors : (n, k) ndarray
        Corresponding eigenvectors.
    n_iter : int
        Number of iterations.
    history : list of (k,) ndarrays
        Eigenvalue estimates per iteration.
    """
    # Dense vs sparse
    if issparse(A):
        n = A.shape[0]
        I = sparse_eye(n, format="csr")
        # estimate largest eigenvalue via ARPACK
        w_max, _ = eigsh(A, k=1, which="LA")
        alpha = float(w_max[0])
        A_mat = A
    else:
        A_mat = np.asarray(A)
        n = A_mat.shape[0]
        I = np.eye(n)
        # exact largest eigenvalue for small dense A
        w_all = eigh(A_mat, eigvals_only=True)
        alpha = float(w_all[-1])

    # 1) Define B = alpha I - A
    B = alpha * I - A_mat

    # 2) Choose working subspace dimension
    if block_size is None:
        block_size = k
    # safety: can't exceed n
    block_size = min(block_size, n)

    # 3) Init Q
    Q = np.random.randn(n, block_size)
    Q, _ = qr(Q, mode="economic")

    eigenvalues_old = None
    history = []

    for it in range(max_iter):
        # Subspace iteration on B
        Q_new = B @ Q
        Q, _ = qr(Q_new, mode="economic")

        # Rayleigh–Ritz on A
        A_proj = Q.T @ (A_mat @ Q)
        evals_proj, evecs_proj = eigh(A_proj)

        # sort ascending
        idx = np.argsort(evals_proj)
        evals_proj = evals_proj[idx]
        evecs_proj = evecs_proj[:, idx]

        # refine Q
        Q = Q @ evecs_proj  # n x block_size

        # extract the k eigenvalues we care about
        if skip_trivial:
            # assume the very smallest is trivial
            eigenvalues = evals_proj[1 : k + 1]
        else:
            eigenvalues = evals_proj[:k]

        history.append(eigenvalues.copy())

        # convergence check
        if eigenvalues_old is not None:
            err = np.max(np.abs(eigenvalues - eigenvalues_old))
            if err < tol:
                break

        eigenvalues_old = eigenvalues.copy()

    # final eigenvectors
    if skip_trivial:
        eigenvectors = Q[:, 1 : k + 1]
    else:
        eigenvectors = Q[:, :k]

    return eigenvalues, eigenvectors, it + 1, history

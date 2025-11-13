"""
Implicitly Restarted Lanczos method (IRL)
----------------------------------------

Computes the k smallest eigenvalues/eigenvectors of a matrix A
using the Implicitly Restarted Lanczos algorithm (Lanczos + implicit QR).

This is a simplified ARPACK-style implementation.

Returns:
    eigenvalues, eigenvectors, n_iter, history
"""

import numpy as np
from scipy.linalg import eigh


def lanczos_step(A, v0, m, tol=1e-14):
    """
    Perform m steps of Lanczos starting from initial vector v0.
    Returns the basis V (n×m), and the tridiagonal matrix data (alpha, beta).
    """
    n = A.shape[0]
    V = np.zeros((n, m))
    alpha = np.zeros(m)
    beta = np.zeros(m - 1)

    v = v0 / np.linalg.norm(v0)
    V[:, 0] = v
    w = A @ v
    alpha[0] = v @ w
    w -= alpha[0] * v

    for j in range(1, m):
        beta[j - 1] = np.linalg.norm(w)
        if beta[j - 1] < tol:
            # breakdown → restart
            v = np.random.randn(n)
            v /= np.linalg.norm(v)
            V[:, j] = v
            w = A @ v
            alpha[j] = v @ w
            w = w - alpha[j] * v
            continue

        v = w / beta[j - 1]
        V[:, j] = v
        w = A @ v
        alpha[j] = v @ w
        w -= alpha[j] * v + beta[j - 1] * V[:, j-1]

    return V, alpha, beta


def build_tridiagonal(alpha, beta):
    """Return the symmetric tridiagonal matrix from alpha/beta."""
    m = len(alpha)
    T = np.zeros((m, m))
    for i in range(m):
        T[i, i] = alpha[i]
        if i < m - 1:
            T[i, i + 1] = beta[i]
            T[i + 1, i] = beta[i]
    return T


def implicit_qr_shift(T, shifts):
    """
    Apply implicit QR steps to T with given shifts (Ritz values).
    Returns the transformed tridiagonal matrix.
    """
    T_mod = T.copy()
    m = T_mod.shape[0]

    for mu in shifts:
        # perform a single implicit QR step with shift mu
        Q, R = np.linalg.qr(T_mod - mu * np.eye(m))
        T_mod = R @ Q + mu * np.eye(m)

        # force symmetry (QR drifts slightly)
        T_mod = 0.5 * (T_mod + T_mod.T)

    return T_mod


def lanczos_implicitqr(
    A, k, m=None, max_outer=50, tol=1e-8, verbose=False
):
    """
    Implicitly Restarted Lanczos algorithm.

    Parameters
    ----------
    A : (n,n) ndarray or sparse matrix
    k : int
        number of eigenvalues to compute
    m : int
        Lanczos subspace dimension (m > k), default = k + 10
    max_outer : int
        number of outer restart cycles
    tol : float
        tolerance on Ritz residuals

    Returns
    -------
    eigenvalues, eigenvectors, n_iter, history
    """

    n = A.shape[0]
    if m is None:
        m = min(n, k + 10)

    # initial vector
    v0 = np.random.randn(n)
    v0 /= np.linalg.norm(v0)

    history = []

    for outer in range(max_outer):

        # -------------------------------
        # 1) Lanczos build of size m
        # -------------------------------
        V, alpha, beta = lanczos_step(A, v0, m)
        T = build_tridiagonal(alpha, beta)

        # -------------------------------
        # 2) Compute Ritz pairs of T
        # -------------------------------
        evals_T, evecs_T = eigh(T)
        idx = np.argsort(evals_T)
        evals_T = evals_T[idx]
        evecs_T = evecs_T[:, idx]

        # Ritz vectors in original space
        ritz_vecs = V @ evecs_T

        # -------------------------------
        # 3) Convergence check
        # -------------------------------
        # compute residual norm || A v - lambda v ||
        ritz_resids = []
        for i in range(k):
            lam = evals_T[i]
            z = ritz_vecs[:, i]
            r = A @ z - lam * z
            ritz_resids.append(np.linalg.norm(r))

        history.append(ritz_resids.copy())

        if verbose:
            print(f"[IRL] outer={outer}, residual={max(ritz_resids):.3e}")

        if max(ritz_resids) < tol:
            # converged
            return (
                evals_T[:k],
                ritz_vecs[:, :k],
                outer + 1,
                history,
            )

        # -------------------------------
        # 4) Choose shifts = unwanted Ritz values
        # -------------------------------
        shifts = evals_T[k:m]   # heuristic: use the "bad" eigenvalues

        # -------------------------------
        # 5) Apply implicit QR shifts to T
        # -------------------------------
        T_new = implicit_qr_shift(T, shifts)

        # update the starting vector (last column of the new Q)
        # because the last Lanczos vector is the new restart vector
        _, evecs_new = eigh(T_new)
        q_last = evecs_new[:, -1]
        v0 = V @ q_last
        v0 /= np.linalg.norm(v0)

    # No convergence
    return evals_T[:k], ritz_vecs[:, :k], max_outer, history

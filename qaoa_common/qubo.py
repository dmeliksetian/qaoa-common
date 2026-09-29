"""
QUBO matrix builder, brute-force solver, and synthetic portfolio instances.
"""
from __future__ import annotations

import numpy as np
from itertools import combinations


def build_qubo(mu: np.ndarray, sigma: np.ndarray, lam: float = 0.5, k: int = 3,
               normalize: bool = False) -> np.ndarray:
    """
    Build a QUBO matrix for the portfolio optimisation problem.

    Minimise  x^T Q x  where x ∈ {0,1}^n subject to sum(x) = k.

    Convention (symmetric Q, evaluated via x^T Q x):
      Q[i,i] = -mu[i] + lam*sigma[i,i] + P*(1 - 2k)
      Q[i,j] = lam*sigma[i,j] + P          (i ≠ j)

    The off-diagonal is P (not 2P) because x^T Q x counts each pair (i,j) and
    (j,i) separately, so the effective penalty per pair is 2*Q[i,j] = 2P.

    Penalty:
      P = sum|mu| + lam*sum|sigma|  — dominates the objective for any k violation.

    Args:
        mu:        Annualised expected returns, shape (n,).
        sigma:     Annualised covariance matrix (Ledoit-Wolf recommended), shape (n,n).
        lam:       Risk aversion parameter (default 0.5).
        k:         Number of assets to select.
        normalize: If True, divide Q by max(|Q|) so entries lie in [-1, 1].
                   Use before hardware runs so gate angles scale to O(1) and
                   noise effects (~0.01–0.1) become visible in the landscape.

    Returns:
        Q: (n x n) symmetric QUBO matrix.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    n = len(mu)

    P = np.abs(mu).sum() + lam * np.abs(sigma).sum()

    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -mu[i] + lam * sigma[i, i] + P * (1 - 2 * k)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = lam * sigma[i, j] + P
            Q[j, i] = Q[i, j]

    if normalize:
        Q = Q / np.abs(Q).max()

    return Q


def brute_force_optimal(Q: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    """
    Exact brute-force solver over all C(n, k) feasible portfolios.

    Args:
        Q: (n x n) QUBO matrix.
        k: Number of assets to select.

    Returns:
        (best_energy, best_config) — energy and binary vector of the optimum.
    """
    n = len(Q)
    best_energy = float('inf')
    best_config = None

    for indices in combinations(range(n), k):
        x = np.zeros(n, dtype=int)
        x[list(indices)] = 1
        energy = x @ Q @ x
        if energy < best_energy:
            best_energy = energy
            best_config = x.copy()

    return best_energy, best_config


def synthetic_instance(seed: int = 42, n: int = 6, k: int = 3, lam: float = 0.5,
                       normalize: bool = False):
    """
    Generate a reproducible synthetic portfolio instance (no yfinance required).

    Returns:
        Q:       (n x n) QUBO matrix (normalized if normalize=True).
        mu:      Annualised returns, shape (n,).
        sigma:   Annualised covariance matrix, shape (n,n).
        tickers: List of synthetic ticker names.
    """
    rng = np.random.default_rng(seed)
    tickers = [f'A{i}' for i in range(n)]

    mu = rng.uniform(0.05, 0.35, size=n)

    # Random positive-definite covariance
    L = rng.standard_normal((n, n)) * 0.1
    sigma = L @ L.T + np.diag(rng.uniform(0.02, 0.08, size=n))

    Q = build_qubo(mu, sigma, lam=lam, k=k, normalize=normalize)
    return Q, mu, sigma, tickers

"""
Classical benchmark pricing: Black--Scholes closed form and a vectorised,
antithetic-variate geometric Brownian motion (GBM) Monte Carlo engine.

These routines provide the ground-truth / classical-Monte-Carlo baseline
against which every quantum and quantum-inspired amplitude-estimation
method in :mod:`qcmc.experiments` is benchmarked (see manuscript
Eq. ``bs_formula`` and Sec. "Classical Monte Carlo baseline").
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "bs_call_price",
    "bs_asian_geometric_price",
    "simulate_gbm_paths",
    "european_call_payoff",
    "asian_call_payoff",
    "price_european_ground_truth",
    "price_asian_ground_truth",
]


def bs_call_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Closed-form Black--Scholes price of a European call option.

    .. math:: V_{BS} = S_0\\,\\Phi(d_1) - K e^{-rT}\\,\\Phi(d_2), \\qquad
              d_{1,2} = \\frac{\\ln(S_0/K) + (r \\pm \\tfrac12\\sigma^2)T}{\\sigma\\sqrt T}.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_asian_geometric_price(
    S0: float, K: float, r: float, sigma: float, T: float, n_steps: int
) -> float:
    """Closed-form price of a *geometric*-average Asian call.

    Used as the analytic control-variate anchor for the arithmetic-average
    Asian option (the geometric mean is log-normal, so a Black-Scholes-style
    closed form exists with a variance-and-drift-adjusted volatility
    ``sigma_g``).
    """
    sigma_g = sigma * np.sqrt((2 * n_steps + 1) / (6 * (n_steps + 1)))
    mu_g = (r - 0.5 * sigma**2) * (n_steps + 1) / (2 * n_steps) + 0.5 * sigma_g**2
    d1 = (np.log(S0 / K) + (mu_g + 0.5 * sigma_g**2) * T) / (sigma_g * np.sqrt(T))
    d2 = d1 - sigma_g * np.sqrt(T)
    return np.exp(-r * T) * (S0 * np.exp(mu_g * T) * norm.cdf(d1) - K * norm.cdf(d2))


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool = True,
) -> np.ndarray:
    """Vectorised GBM path simulation under the risk-neutral measure.

    Parameters
    ----------
    antithetic : bool
        If True, draws ``n_paths // 2`` standard normals and mirrors them
        (``z`` and ``-z``) as an elementary variance-reduction technique.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, n_steps + 1)``, including :math:`S_0` at
        column 0.
    """
    dt = T / n_steps
    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        z = np.vstack([z, -z])
        n_paths_eff = z.shape[0]
    else:
        z = rng.standard_normal((n_paths, n_steps))
        n_paths_eff = n_paths
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    S = S0 * np.exp(log_paths)
    S = np.hstack([np.full((n_paths_eff, 1), S0), S])
    return S


def european_call_payoff(S_T: np.ndarray, K: float) -> np.ndarray:
    """Vanilla European call payoff :math:`\\max(S_T - K, 0)`."""
    return np.maximum(S_T - K, 0.0)


def asian_call_payoff(paths: np.ndarray, K: float) -> np.ndarray:
    """Arithmetic-average Asian call payoff.

    Averages the monitoring dates (columns ``1..n_steps``, excluding
    :math:`S_0` in column 0).
    """
    avg = paths[:, 1:].mean(axis=1)
    return np.maximum(avg - K, 0.0)


def price_european_ground_truth(params: dict) -> float:
    """Ground-truth European call price via the closed-form Black-Scholes
    formula. ``params`` must contain the keys ``S0, K, r, sigma, T``."""
    return bs_call_price(**params)


def price_asian_ground_truth(
    params: dict, n_steps: int = 12, n_paths: int = 4_000_000, seed: int = 20260906 + 1
):
    """High-fidelity classical Monte Carlo ground truth for the arithmetic
    Asian call (no closed form exists), returning ``(price, standard_error)``.
    """
    rng_gt = np.random.default_rng(seed)
    paths = simulate_gbm_paths(
        params["S0"], params["r"], params["sigma"], params["T"], n_steps, n_paths, rng_gt
    )
    payoff = asian_call_payoff(paths, params["K"])
    disc_payoff = np.exp(-params["r"] * params["T"]) * payoff
    return disc_payoff.mean(), disc_payoff.std(ddof=1) / np.sqrt(len(disc_payoff))

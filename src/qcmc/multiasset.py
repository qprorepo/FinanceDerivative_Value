"""
Multi-asset basket-option pricing via a quantum Cholesky-entanglement
circuit (manuscript Sec. "Multi-Asset Derivatives", Eqs.
``multiasset_amp``, ``cholesky_general``).

The joint :math:`K`-asset terminal-price statevector is built by applying
the Cholesky entangling map :math:`\\bm z' = \\bm L \\bm z`,
:math:`\\bm L \\bm L^\\top = \\bm\\Sigma`, to :math:`K` independently
discretised standard-normal registers, giving the *exact* discrete joint
amplitude -- no Monte Carlo noise contaminates the resulting amplitude
itself; all statistical error enters only through the downstream Grover
measurement simulation (see :mod:`qcmc.experiments`).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "build_correlated_multiasset_state",
    "basket_call_price_quantum",
    "basket_amplitude_and_scale",
]


def build_correlated_multiasset_state(
    K: int,
    n_qubits_per_asset: int,
    Sigma: np.ndarray,
    S0_vec,
    r: float,
    T: float,
    n_std: float = 3.5,
) -> dict:
    """Constructs the full :math:`2^{Kn}`-dimensional amplitude array of a
    :math:`K`-asset joint log-normal distribution, correlated via the
    Cholesky factor :math:`\\bm L` of ``Sigma``:

    .. math::
        |\\Psi\\rangle = \\sum_{z_1,\\dots,z_K}
        \\sqrt{\\varphi(z_1)\\cdots\\varphi(z_K)}\\,
        |x_1(\\bm L \\bm z)\\rangle \\cdots |x_K(\\bm L \\bm z)\\rangle,

    where :math:`\\bm z' = \\bm L \\bm z` is the correlated standard-normal
    vector and :math:`x_j(\\bm z')` is the discretised log-price bin index
    implied by the :math:`j`-th correlated normal factor.

    Returns a dict with keys ``S_paths`` (shape ``(K, n_bins**K)``, the
    terminal price realised at each joint grid point for each asset),
    ``joint_prob`` (shape ``(n_bins**K,)``), ``L``, ``sigma_j``, ``n_bins``,
    ``K``.
    """
    L = np.linalg.cholesky(Sigma)
    sigma_j = np.sqrt(np.diag(Sigma))
    n_bins = 2**n_qubits_per_asset

    z_edges = np.linspace(-n_std, n_std, n_bins + 1)
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    z_prob_1d = np.diff(norm.cdf(z_edges))
    z_prob_1d = z_prob_1d / z_prob_1d.sum()

    mesh = np.meshgrid(*([z_centres] * K), indexing="ij")
    Z = np.stack([m.ravel() for m in mesh], axis=0)
    joint_prob = np.ones(Z.shape[1])
    for j in range(K):
        idx_j = np.unravel_index(np.arange(Z.shape[1]), (n_bins,) * K)[j]
        joint_prob *= z_prob_1d[idx_j]

    Zc = L @ Z

    mu_ln = np.log(np.asarray(S0_vec)) + (r - 0.5 * sigma_j**2) * T
    # NOTE: Zc already has the correct marginal standard deviation sigma_j
    # baked in via the Cholesky factor of the *covariance* matrix Sigma
    # (Cov(L @ Z) = L L^T = Sigma exactly, since Z ~ iid N(0, I)), so no
    # further multiplication by sigma_j is applied here. An earlier version
    # of this function multiplied by `sigma_j` again, which silently
    # replaced the intended sigma_j with sigma_j**2 in the exponent —
    # a severe (and easy to miss) volatility-understatement bug for any
    # sigma_j < 1, caught by
    # ``tests/test_multiasset.py::test_single_asset_basket_reduces_to_vanilla_call``.
    S_paths = np.exp(mu_ln[:, None] + np.sqrt(T) * Zc)

    return dict(S_paths=S_paths, joint_prob=joint_prob, L=L, sigma_j=sigma_j, n_bins=n_bins, K=K)


def basket_call_price_quantum(
    state: dict, weights: np.ndarray, K_strike: float, r: float, T: float
) -> float:
    """:math:`\\mathbb E^{\\mathbb Q}[\\max(\\bm w^\\top \\bm S_T - K, 0)]`,
    evaluated exactly on the discretised joint distribution built by
    :func:`build_correlated_multiasset_state` (Eq.
    ``multiasset_amp_identity`` with :math:`f` the basket payoff).
    """
    basket_value = weights @ state["S_paths"]
    payoff = np.maximum(basket_value - K_strike, 0.0)
    price = np.exp(-r * T) * float(np.sum(state["joint_prob"] * payoff))
    return price


def basket_amplitude_and_scale(
    K: int,
    sigma_5factor: np.ndarray,
    corr_5factor: np.ndarray,
    r: float,
    n_q_per_asset: int | None = None,
    vol_target: float = 0.22,
    max_dim: int = 2_000_000,
    seed: int = 20260906,
):
    """Convenience wrapper: builds a :math:`K`-asset equal-weight basket
    call, correlated by the empirical 5-factor correlation block (extended
    with weakly-correlated synthetic assets for :math:`K>5`), and returns
    ``(amplitude, price_scale)`` such that
    ``price = amplitude * price_scale``.

    The number of qubits per asset is automatically shrunk as :math:`K`
    grows (``n_qubits_per_asset = floor(log2(max_dim) / K)``, floored at 2)
    so the full :math:`K`-register tensor-product grid stays memory
    tractable.

    Parameters
    ----------
    sigma_5factor : np.ndarray, shape (5, 5)
        Annualised covariance of the five Fama-French factors (unused
        directly; only its *correlation* structure via ``corr_5factor`` is
        used, rescaled to ``vol_target``).
    corr_5factor : np.ndarray, shape (5, 5)
        Empirical correlation matrix of the five Fama-French factors.
    """
    if n_q_per_asset is None:
        n_q_per_asset = max(2, int(np.floor(np.log2(max_dim) / K)))

    if K <= 5:
        Corr_K = corr_5factor[:K, :K]
    else:
        base = np.eye(K)
        base[:5, :5] = corr_5factor
        extra_corr = 0.15
        for i in range(5, K):
            base[i, :i] = base[:i, i] = extra_corr
        Corr_K = base

    Sigma_scaled = (vol_target**2) * Corr_K

    state = build_correlated_multiasset_state(
        K, n_q_per_asset, Sigma_scaled, S0_vec=[100] * K, r=r, T=1.0
    )
    weights = np.full(K, 1.0 / K)
    basket_val = weights @ state["S_paths"]
    payoff_norm = np.clip(np.maximum(basket_val - 100, 0.0) / basket_val.max(), 0, 1)
    a_basket = float(np.sum(state["joint_prob"] * payoff_norm))
    price_scale = np.exp(-r * 1.0) * basket_val.max()
    return a_basket, price_scale

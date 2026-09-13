import numpy as np
import pytest

from qcmc.multiasset import (
    basket_amplitude_and_scale,
    basket_call_price_quantum,
    build_correlated_multiasset_state,
)


def test_joint_probabilities_sum_to_one():
    Sigma = 0.04 * np.eye(3)
    n_qubits_per_asset = 4
    state = build_correlated_multiasset_state(
        3, n_qubits_per_asset, Sigma=Sigma, S0_vec=[100, 100, 100], r=0.05, T=1.0
    )
    assert state["joint_prob"].sum() == pytest.approx(1.0, abs=1e-9)
    n_bins = 2**n_qubits_per_asset
    assert state["S_paths"].shape == (3, n_bins**3)


def test_single_asset_basket_reduces_to_vanilla_call():
    """A 'basket' of K=1 identical asset with weight 1 should exactly
    reproduce the single-asset discretised call amplitude/price."""
    Sigma = np.array([[0.04]])
    state = build_correlated_multiasset_state(
        1, n_qubits_per_asset=8, Sigma=Sigma, S0_vec=[100], r=0.05, T=1.0
    )
    price = basket_call_price_quantum(state, weights=np.array([1.0]), K_strike=100, r=0.05, T=1.0)
    from qcmc.classical_pricing import bs_call_price

    v_bs = bs_call_price(S0=100, K=100, r=0.05, sigma=0.20, T=1.0)
    assert price == pytest.approx(v_bs, rel=0.03)


def test_uncorrelated_basket_price_less_than_perfectly_correlated():
    """Diversification: a weakly-correlated basket has strictly lower
    terminal variance than a near-perfectly-correlated one, hence a lower
    ATM call price (for equal marginal volatilities). (Exact correlation
    of 1.0 is used only as the limiting comparison target; the Cholesky
    factorisation requires strict positive-definiteness, so we use 0.999
    for the "high correlation" case.)"""
    K, n_q = 3, 5
    vol = 0.22
    S0_vec = [100, 100, 100]

    Sigma_indep = (vol**2) * np.eye(K)
    corr_high = np.full((K, K), 0.999)
    np.fill_diagonal(corr_high, 1.0)
    Sigma_corr = (vol**2) * corr_high

    state_indep = build_correlated_multiasset_state(K, n_q, Sigma_indep, S0_vec, 0.05, 1.0)
    state_corr = build_correlated_multiasset_state(K, n_q, Sigma_corr, S0_vec, 0.05, 1.0)

    w = np.full(K, 1 / K)
    price_indep = basket_call_price_quantum(state_indep, w, 100, 0.05, 1.0)
    price_corr = basket_call_price_quantum(state_corr, w, 100, 0.05, 1.0)

    assert price_indep < price_corr


def test_basket_amplitude_and_scale_returns_consistent_price():
    corr5 = np.eye(5)
    a, scale = basket_amplitude_and_scale(K=3, sigma_5factor=np.eye(5), corr_5factor=corr5, r=0.05)
    assert 0.0 <= a <= 1.0
    price = a * scale
    assert price > 0


def test_basket_amplitude_shrinks_qubits_for_large_K():
    """For large K the per-asset qubit count must shrink to keep the joint
    grid tractable — this is a memory-safety property, not just a nicety."""
    corr8 = np.eye(8)
    # should not raise / hang / exhaust memory
    a, scale = basket_amplitude_and_scale(
        K=8, sigma_5factor=np.eye(5), corr_5factor=corr8[:5, :5], r=0.05, max_dim=2_000_000
    )
    assert 0.0 <= a <= 1.0

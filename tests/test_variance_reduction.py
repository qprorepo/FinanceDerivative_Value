import numpy as np
import pytest

from qcmc.classical_pricing import european_call_payoff, simulate_gbm_paths
from qcmc.quantum_ae import discretise_lognormal
from qcmc.variance_reduction import (
    calibrate_tilt_for_target_amplitude,
    dual_control_variate_stats,
    empirical_control_variate_stats,
    exponential_tilt,
    importance_sampling_speedup,
)


def test_control_variate_perfect_correlation_gives_infinite_speedup():
    """If payoff == control exactly, rho=1 and R_CV -> infinity (division
    by ~0, which is the mathematically correct — not erroneous — behaviour
    of Eq. cv_speedup at that limit); we just check R_CV explodes rather
    than staying near 1."""
    rng = np.random.default_rng(0)
    f = rng.normal(10, 2, size=10_000)
    w = f.copy()
    with np.errstate(divide="ignore"):
        stats = empirical_control_variate_stats(f, w)
    assert stats["rho"] == pytest.approx(1.0, abs=1e-9)
    assert stats["R_CV"] > 1e6


def test_control_variate_zero_correlation_gives_no_speedup():
    rng = np.random.default_rng(0)
    f = rng.normal(10, 2, size=200_000)
    w = rng.normal(5, 1, size=200_000)  # independent of f
    stats = empirical_control_variate_stats(f, w)
    assert abs(stats["rho"]) < 0.02
    assert stats["R_CV"] == pytest.approx(1.0, abs=0.05)


def test_control_variate_matches_manuscript_european_call_scenario(canonical_bs_params):
    p = canonical_bs_params
    rng = np.random.default_rng(20260906 + 21)
    paths = simulate_gbm_paths(p["S0"], p["r"], p["sigma"], p["T"], 1, 400_000, rng)
    S_T = paths[:, -1]
    disc = np.exp(-p["r"] * p["T"])
    f = disc * european_call_payoff(S_T, p["K"])
    w = disc * S_T
    stats = empirical_control_variate_stats(f, w)
    # manuscript reports rho_CV = 0.9245, R_CV = 6.89x on this exact scenario
    assert stats["rho"] == pytest.approx(0.9245, abs=0.02)
    assert stats["R_CV"] == pytest.approx(6.89, rel=0.15)


def test_dual_control_variate_reduces_to_single_cv_with_one_control():
    rng = np.random.default_rng(1)
    f = rng.normal(size=50_000) + 0.8 * rng.normal(size=50_000)
    w = f + rng.normal(scale=0.3, size=50_000)
    single = empirical_control_variate_stats(f, w)
    dual = dual_control_variate_stats(f, w.reshape(-1, 1))
    assert dual["quad_form"] == pytest.approx(single["rho"] ** 2, abs=1e-6)
    assert dual["R_CV"] == pytest.approx(single["R_CV"], rel=1e-4)


def test_exponential_tilt_produces_valid_probability_distribution(canonical_bs_params):
    p = canonical_bs_params
    dist = discretise_lognormal(p["S0"], p["r"], p["sigma"], p["T"], n_qubits=6)
    payoff = lambda s: european_call_payoff(s, p["K"]) / (dist.x_grid.max() - p["K"])
    q = exponential_tilt(dist, payoff, eta=5.0)
    assert q.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(q >= 0)


def test_exponential_tilt_with_eta_zero_recovers_original_distribution(canonical_bs_params):
    p = canonical_bs_params
    dist = discretise_lognormal(p["S0"], p["r"], p["sigma"], p["T"], n_qubits=6)
    payoff = lambda s: european_call_payoff(s, p["K"]) / (dist.x_grid.max() - p["K"])
    q = exponential_tilt(dist, payoff, eta=0.0)
    np.testing.assert_allclose(q, dist.probs, atol=1e-10)


def test_calibrate_tilt_hits_target_amplitude(canonical_bs_params):
    p = canonical_bs_params
    dist = discretise_lognormal(p["S0"], p["r"], p["sigma"], p["T"], n_qubits=6)
    payoff = lambda s: european_call_payoff(s, p["K"]) / (dist.x_grid.max() - p["K"])
    eta = calibrate_tilt_for_target_amplitude(dist, payoff, target_B=0.25)
    q = exponential_tilt(dist, payoff, eta)
    f_vals = np.clip(payoff(dist.x_grid), 0, 1)
    achieved = float(np.sum(q * f_vals))
    assert achieved == pytest.approx(0.25, abs=1e-3)


def test_importance_sampling_speedup_formula():
    assert importance_sampling_speedup(a=0.01, B=0.25) == pytest.approx(5.0)
    assert importance_sampling_speedup(a=0.25, B=0.25) == pytest.approx(1.0)


def test_importance_sampling_speedup_larger_for_smaller_amplitude():
    """R_IS = sqrt(B/a) should be monotonically decreasing in a — deeper
    out-of-the-money / tail-risk payoffs (small a) benefit more from IS."""
    r_small_a = importance_sampling_speedup(a=0.001, B=0.2)
    r_large_a = importance_sampling_speedup(a=0.1, B=0.2)
    assert r_small_a > r_large_a

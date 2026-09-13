import numpy as np
import pytest

from qcmc.classical_pricing import bs_call_price, european_call_payoff
from qcmc.quantum_ae import (
    amplitude_from_payoff,
    classical_fisher_info,
    discretise_lognormal,
    ideal_grover_prob,
    noisy_ancilla_prob,
    optimal_grover_depth,
    sample_ancilla_measurements,
    theta_from_amplitude,
)


def test_discretised_distribution_probabilities_sum_to_one(canonical_bs_params):
    p = canonical_bs_params
    dist = discretise_lognormal(p["S0"], p["r"], p["sigma"], p["T"], n_qubits=6)
    assert dist.probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert dist.n_bins == 64
    assert np.all(dist.probs >= 0)


def test_amplitude_from_payoff_is_bounded_in_unit_interval(canonical_bs_params):
    p = canonical_bs_params
    dist = discretise_lognormal(p["S0"], p["r"], p["sigma"], p["T"], n_qubits=6)
    payoff = lambda s: european_call_payoff(s, p["K"]) / (dist.x_grid.max() - p["K"])
    a = amplitude_from_payoff(dist, payoff)
    assert 0.0 <= a <= 1.0


def test_amplitude_reproduces_black_scholes_price(canonical_bs_params):
    """The n=6-qubit discretised amplitude, rescaled, should approximate
    the closed-form Black-Scholes price to within its known discretisation
    bias (a few percent at this coarse qubit count)."""
    p = canonical_bs_params
    dist = discretise_lognormal(p["S0"], p["r"], p["sigma"], p["T"], n_qubits=6)
    scale = dist.x_grid.max() - p["K"]
    payoff = lambda s: european_call_payoff(s, p["K"]) / scale
    a = amplitude_from_payoff(dist, payoff)
    disc = np.exp(-p["r"] * p["T"])
    price_disc = a * scale * disc
    price_bs = bs_call_price(**p)
    assert price_disc == pytest.approx(price_bs, rel=0.02)


def test_theta_amplitude_round_trip():
    for a in [0.001, 0.05, 0.25, 0.5, 0.9, 0.999]:
        theta = theta_from_amplitude(a)
        assert np.sin(theta) ** 2 == pytest.approx(a, abs=1e-9)


def test_ideal_grover_prob_at_m_zero_is_amplitude():
    """At m=0, no Grover iterations have been applied, so
    P = sin^2(theta_A) = a exactly."""
    theta = theta_from_amplitude(0.3)
    p0 = ideal_grover_prob(theta, 0)
    assert p0 == pytest.approx(0.3, abs=1e-9)


def test_noisy_ancilla_prob_reduces_to_ideal_grover_in_noiseless_limit():
    theta = 0.35
    m = np.arange(0, 20)
    p_noisy = noisy_ancilla_prob(theta, m, gamma_d=0.0, fidelity=1.0)
    p_ideal = ideal_grover_prob(theta, m)
    np.testing.assert_allclose(p_noisy, p_ideal, atol=1e-10)


def test_noisy_ancilla_prob_collapses_to_half_at_high_depth_with_noise():
    """As gamma_D * m -> infinity, the ancilla decoheres to the maximally
    mixed state, giving P(D=1) -> 1/2 regardless of theta_A."""
    theta = 0.4
    p = noisy_ancilla_prob(theta, m=10_000, gamma_d=0.01, fidelity=0.99)
    assert p == pytest.approx(0.5, abs=1e-6)


def test_classical_fisher_info_noiseless_limit_matches_ideal_scaling():
    """As gamma_D -> 0, fidelity -> 1, I_m(theta) -> (2m+1)^2 exactly,
    independent of theta (an algebraic identity used throughout the
    manuscript's Cramer-Rao bound discussion)."""
    theta = 0.4
    m = np.arange(1, 40)
    I_noisy = classical_fisher_info(theta, m, gamma_d=1e-9, fidelity=1 - 1e-9)
    I_ideal = (2 * m + 1) ** 2
    max_dev = np.max(np.abs(I_noisy / I_ideal - 1))
    assert max_dev < 1e-3


@pytest.mark.parametrize("theta", [0.1, 0.3, 0.5, 0.7, 1.0])
def test_classical_fisher_info_nonnegative(theta):
    m = np.arange(0, 30)
    fisher_info = classical_fisher_info(theta, m, gamma_d=1e-3, fidelity=0.999)
    assert np.all(fisher_info >= -1e-9)


def test_optimal_grover_depth_matches_manuscript_worked_example():
    """The manuscript explicitly states m* ~ 250 for F=0.999, gamma_D=1e-3."""
    m_star = optimal_grover_depth(gamma_d=1e-3, fidelity=0.999)
    assert m_star == pytest.approx(250, rel=0.05)


def test_optimal_grover_depth_decreases_with_more_noise():
    m_low_noise = optimal_grover_depth(gamma_d=1e-3, fidelity=0.999)
    m_high_noise = optimal_grover_depth(gamma_d=1e-2, fidelity=0.990)
    assert m_high_noise < m_low_noise


def test_sample_ancilla_measurements_shapes_and_law_of_large_numbers(rng):
    theta = 0.3
    m_seq = np.array([0, 1, 2, 4, 8])
    shots = 100_000
    m_out, n_ones, n_shots = sample_ancilla_measurements(
        theta, m_seq, gamma_d=0.0, fidelity=1.0, shots_per_depth=shots, rng=rng
    )
    assert n_ones.shape == m_seq.shape
    assert np.all(n_shots == shots)
    empirical_p = n_ones / shots
    expected_p = ideal_grover_prob(theta, m_seq)
    np.testing.assert_allclose(empirical_p, expected_p, atol=0.01)

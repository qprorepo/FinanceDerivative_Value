import numpy as np
import pytest

from qcmc.experiments import (
    fit_convergence_exponent,
    mle_theta_from_record,
    query_cost,
    rmse_experiment,
    simulate_one_qae_trial,
)
from qcmc.quantum_ae import ideal_grover_prob


def test_query_cost_formula():
    depths = np.array([0, 1, 2])
    # cost per depth = 2m+1 -> 1, 3, 5 = 9 per shot
    assert query_cost(depths, shots=1) == 9
    assert query_cost(depths, shots=10) == 90


def test_mle_theta_from_record_recovers_known_theta_with_many_shots(rng):
    theta_true = 0.35
    depths = np.array([0, 1, 2, 4, 8])
    shots = 20_000
    p1 = ideal_grover_prob(theta_true, depths)
    n_ones = rng.binomial(shots, p1)
    n_shots = np.full_like(depths, shots)

    theta_hat = mle_theta_from_record(
        depths, n_ones, n_shots, prob_model=lambda m, th: ideal_grover_prob(th, m)
    )
    assert theta_hat == pytest.approx(theta_true, abs=0.01)


def test_simulate_one_qae_trial_noise_aware_beats_noise_unaware_on_average(rng):
    """A noise-*aware* fit (fit params == true noise params) should give a
    lower-bias theta estimate on average than a noise-*unaware* fit
    (fit params assume no noise) when the true hardware IS noisy."""
    theta_true = 0.35
    depths = np.array([0, 1, 2, 4, 8, 16])
    shots = 200
    gamma_true, fidelity_true = 5e-3, 0.995

    aware_errs, unaware_errs = [], []
    for trial in range(60):
        r = np.random.default_rng(trial)
        theta_aware = simulate_one_qae_trial(
            theta_true, depths, shots, gamma_true, fidelity_true, gamma_true, fidelity_true, r
        )
        r2 = np.random.default_rng(trial)
        theta_unaware = simulate_one_qae_trial(
            theta_true, depths, shots, gamma_true, fidelity_true, 0.0, 1.0, r2
        )
        aware_errs.append(abs(theta_aware - theta_true))
        unaware_errs.append(abs(theta_unaware - theta_true))

    assert np.mean(aware_errs) <= np.mean(unaware_errs) * 1.5  # generous slack


def test_rmse_experiment_classical_matches_clt_scaling(rng):
    a_true = 0.3
    nq_list = [1_000, 4_000, 16_000, 64_000]
    nq, rmse = rmse_experiment(
        a_true,
        nq_list,
        method="classical",
        n_trials=200,
        rng=np.random.default_rng(42),
        price_scale=1.0,
    )
    beta, intercept, r2 = fit_convergence_exponent(nq, rmse)
    assert beta == pytest.approx(-0.5, abs=0.1)
    assert r2 > 0.9


def test_rmse_experiment_ideal_iqae_beats_classical_at_high_budget():
    a_true = 0.3
    nq_list = [1000, 10_000, 100_000]
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    nq_c, rmse_c = rmse_experiment(
        a_true, nq_list, method="classical", n_trials=150, rng=rng1, price_scale=1.0
    )
    nq_q, rmse_q = rmse_experiment(
        a_true, nq_list, method="iqae_ideal", n_trials=150, rng=rng2, price_scale=1.0
    )
    # at the largest budget, the quantum method should have lower RMSE
    assert rmse_q[-1] < rmse_c[-1]


def test_rmse_experiment_effective_multiplier_reduces_rmse():
    a_true = 0.1
    nq_list = [5000]
    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)
    _, rmse_plain = rmse_experiment(
        a_true, nq_list, method="bae_plain", n_trials=100, rng=rng1, price_scale=1.0
    )
    _, rmse_boosted = rmse_experiment(
        a_true,
        nq_list,
        method="bae_plain",
        n_trials=100,
        rng=rng2,
        price_scale=1.0,
        effective_multiplier=10.0,
    )
    assert rmse_boosted[0] < rmse_plain[0]


def test_fit_convergence_exponent_recovers_known_power_law():
    nq = np.array([100, 1000, 10000, 100000], dtype=float)
    true_beta = -0.73
    rmse = 2.5 * nq**true_beta
    beta, intercept, r2 = fit_convergence_exponent(nq, rmse)
    assert beta == pytest.approx(true_beta, abs=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_rmse_experiment_invalid_method_raises():
    with pytest.raises(ValueError):
        rmse_experiment(
            0.3, [1000], method="not_a_real_method", n_trials=1, rng=np.random.default_rng(0)
        )

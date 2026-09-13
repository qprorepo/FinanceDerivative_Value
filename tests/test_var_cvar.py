import numpy as np
import pytest

from qcmc.var_cvar import (
    build_random_portfolio,
    portfolio_loss_samples,
    quantum_binary_search_var,
    quantum_cvar_from_var,
    var_cvar_classical,
)


def test_build_random_portfolio_weights_sum_to_one():
    port = build_random_portfolio(K=10, seed=1)
    assert port["w"].sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(port["w"] >= 0)
    # correlation matrix (implicit in Sigma) must be PSD
    eigvals = np.linalg.eigvalsh(port["Sigma"])
    assert np.all(eigvals >= -1e-8)


def test_var_less_than_or_equal_to_cvar(rng):
    port = build_random_portfolio(K=8, seed=2)
    var_a, cvar_a = var_cvar_classical(port, n_paths=200_000, alpha=0.99, rng=rng)
    assert var_a <= cvar_a  # CVaR is the mean of the tail beyond VaR


def test_higher_confidence_gives_higher_var(rng):
    port = build_random_portfolio(K=8, seed=3)
    var_95, _ = var_cvar_classical(port, n_paths=300_000, alpha=0.95, rng=np.random.default_rng(10))
    var_99, _ = var_cvar_classical(port, n_paths=300_000, alpha=0.99, rng=np.random.default_rng(10))
    assert var_99 > var_95


def test_portfolio_loss_samples_reasonable_scale():
    port = build_random_portfolio(K=5, seed=4)
    losses = portfolio_loss_samples(port, n_paths=100_000, rng=np.random.default_rng(5))
    # losses should be centred roughly around -notional * E[w.mu] (a gain on
    # average, since mu > 0), with a spread set by the portfolio volatility
    assert losses.shape == (100_000,)
    assert abs(losses.mean()) < port["notional"]  # sane order of magnitude


def test_quantum_binary_search_var_converges_near_classical_reference():
    port = build_random_portfolio(K=10, seed=5)
    loss_pool = portfolio_loss_samples(port, n_paths=400_000, rng=np.random.default_rng(6))
    var_ref, _ = var_cvar_classical(port, n_paths=400_000, alpha=0.99, rng=np.random.default_rng(6))

    # average over several trials to reduce the single-draw noise inherent
    # to this simulated (not deterministic) estimator
    estimates = []
    for t in range(20):
        v, nq = quantum_binary_search_var(
            loss_pool,
            alpha=0.99,
            n_shots_per_step=400,
            k_bisection_steps=8,
            rng=np.random.default_rng(100 + t),
        )
        estimates.append(v)
        assert nq == 400 * 8
    mean_estimate = np.mean(estimates)
    assert abs(mean_estimate - var_ref) / abs(var_ref) < 0.15


def test_bayesian_init_reduces_variance_of_var_estimate():
    """The Bayesian posterior-initialised variant should have lower
    across-trial variance than the standard variant at equal query cost."""
    port = build_random_portfolio(K=10, seed=7)
    loss_pool = portfolio_loss_samples(port, n_paths=400_000, rng=np.random.default_rng(8))

    def run(bayesian, n_trials=30):
        vals = []
        for t in range(n_trials):
            v, _ = quantum_binary_search_var(
                loss_pool, 0.99, 200, 5, rng=np.random.default_rng(1000 + t), bayesian_init=bayesian
            )
            vals.append(v)
        return np.var(vals)

    var_standard = run(False)
    var_bayesian = run(True)
    assert var_bayesian <= var_standard * 1.1  # allow small stochastic slack


def test_quantum_cvar_from_var_exceeds_var():
    port = build_random_portfolio(K=6, seed=9)
    loss_pool = portfolio_loss_samples(port, n_paths=200_000, rng=np.random.default_rng(10))
    var_hat = np.percentile(loss_pool, 99)
    cvar_hat = quantum_cvar_from_var(
        loss_pool, var_hat, n_shots=1000, alpha=0.99, rng=np.random.default_rng(11)
    )
    assert cvar_hat >= var_hat - 1e-6

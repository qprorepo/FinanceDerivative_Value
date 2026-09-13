import numpy as np
import pytest

from qcmc.classical_pricing import (
    asian_call_payoff,
    bs_asian_geometric_price,
    bs_call_price,
    european_call_payoff,
    price_asian_ground_truth,
    price_european_ground_truth,
    simulate_gbm_paths,
)


def test_bs_call_price_matches_known_value(canonical_bs_params):
    """Regression test against the textbook value for S0=K=100, r=5%,
    sigma=20%, T=1 (also the value reported throughout the manuscript)."""
    V = bs_call_price(**canonical_bs_params)
    assert V == pytest.approx(10.4506, abs=1e-3)


def test_bs_call_price_put_call_style_bounds(canonical_bs_params):
    """A call price must lie in [max(S0 - K e^{-rT}, 0), S0]."""
    p = canonical_bs_params
    V = bs_call_price(**p)
    lower = max(p["S0"] - p["K"] * np.exp(-p["r"] * p["T"]), 0.0)
    assert lower <= V <= p["S0"]


def test_bs_call_price_deep_itm_converges_to_intrinsic():
    """Deep in-the-money, with tiny volatility, price -> discounted intrinsic value."""
    V = bs_call_price(S0=200, K=100, r=0.05, sigma=1e-4, T=1.0)
    intrinsic_disc = 200 - 100 * np.exp(-0.05)
    assert V == pytest.approx(intrinsic_disc, abs=1e-2)


def test_bs_call_price_deep_otm_near_zero():
    """Deep out-of-the-money, with tiny volatility, price -> 0."""
    V = bs_call_price(S0=50, K=100, r=0.05, sigma=1e-4, T=1.0)
    assert V == pytest.approx(0.0, abs=1e-6)


def test_geometric_asian_lower_than_arithmetic(canonical_bs_params, rng):
    """AM-GM inequality: the geometric mean never exceeds the arithmetic
    mean, so the geometric-average Asian call must be strictly cheaper
    than the arithmetic-average Asian call (for non-degenerate volatility)."""
    p = canonical_bs_params
    n_steps = 12
    v_geo = bs_asian_geometric_price(p["S0"], p["K"], p["r"], p["sigma"], p["T"], n_steps)

    v_arith, se_arith = price_asian_ground_truth(p, n_steps=n_steps, n_paths=200_000, seed=1)
    assert v_geo < v_arith
    # geometric price should still be within a sane multiple of the
    # arithmetic price for these parameters (sanity bound, not a tight one)
    assert 0.5 * v_arith < v_geo < v_arith


def test_simulate_gbm_paths_shape_and_martingale_property(rng):
    """E^Q[S_T] should equal S0 * e^{rT} under the risk-neutral measure
    (the discounted price process is a martingale)."""
    S0, r, sigma, T = 100.0, 0.05, 0.2, 1.0
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps=1, n_paths=2_000_000, rng=rng)
    assert paths.shape == (2_000_000, 2)
    assert paths[:, 0].max() == paths[:, 0].min() == S0

    S_T = paths[:, -1]
    expected = S0 * np.exp(r * T)
    # antithetic variates give a tight martingale check even at 2M paths
    assert S_T.mean() == pytest.approx(expected, rel=5e-3)


def test_european_call_payoff_never_negative():
    S_T = np.array([50.0, 90.0, 100.0, 110.0, 200.0])
    payoff = european_call_payoff(S_T, K=100.0)
    assert np.all(payoff >= 0)
    np.testing.assert_allclose(payoff, [0, 0, 0, 10, 100])


def test_asian_call_payoff_averages_correctly():
    # 3 paths, 2 monitoring steps (+S0 column)
    paths = np.array(
        [
            [100, 110, 120],  # avg of [110,120] = 115 -> payoff 15
            [100, 100, 100],  # avg = 100 -> payoff 0
            [100, 80, 60],  # avg = 70 -> payoff 0
        ]
    )
    payoff = asian_call_payoff(paths, K=100.0)
    np.testing.assert_allclose(payoff, [15, 0, 0])


def test_price_european_ground_truth_matches_direct_call(canonical_bs_params):
    assert price_european_ground_truth(canonical_bs_params) == bs_call_price(**canonical_bs_params)

import numpy as np
import pytest

from qcmc.cat_pricing import (
    cat_excess_payoff,
    discretise_empirical_linear,
    discretise_empirical_log_spaced,
    parse_damage,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.00K", 0.0),
        ("1.00K", 1_000.0),
        ("75.00K", 75_000.0),
        ("1.00M", 1_000_000.0),
        ("2.50M", 2_500_000.0),
        ("1.00B", 1_000_000_000.0),
        ("0.50K", 500.0),
        ("0", 0.0),
        ("100", 100.0),
    ],
)
def test_parse_damage_suffixes(raw, expected):
    assert parse_damage(raw) == pytest.approx(expected)


def test_parse_damage_missing_and_invalid():
    assert np.isnan(parse_damage(None))
    assert np.isnan(parse_damage(float("nan")))
    assert np.isnan(parse_damage(""))
    assert np.isnan(parse_damage("not-a-number"))


def test_discretise_empirical_log_spaced_probabilities_valid(rng):
    samples = rng.lognormal(mean=10, sigma=2, size=10_000)
    dist = discretise_empirical_log_spaced(samples, n_qubits=6)
    assert dist.probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert dist.n_bins == 64
    assert dist.log_spaced is True
    assert np.all(np.diff(dist.x_grid) > 0)  # monotone increasing bin centres


def test_discretise_empirical_linear_probabilities_valid(rng):
    samples = rng.lognormal(mean=10, sigma=2, size=10_000)
    dist = discretise_empirical_linear(samples, n_qubits=6)
    assert dist.probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert dist.log_spaced is False


def test_log_spaced_allocates_more_bins_to_small_losses_than_linear(rng):
    """For a heavy-tailed sample, log-spacing should place a larger share
    of its total bin *count* below the median than linear spacing does
    (even though both have the same 64 bins total) -- this is the
    qualitative property that motivates log-binning for tail risk."""
    samples = rng.lognormal(mean=10, sigma=2.5, size=50_000)
    dist_log = discretise_empirical_log_spaced(samples, n_qubits=6)
    dist_lin = discretise_empirical_linear(samples, n_qubits=6)
    median = np.median(samples)
    frac_log = np.mean(dist_log.x_grid < median)
    frac_lin = np.mean(dist_lin.x_grid < median)
    assert frac_log > frac_lin


def test_cat_excess_payoff_zero_below_attachment():
    x = np.array([0, 1, 4.9, 5.0, 5.1, 100, 1000])
    payoff = cat_excess_payoff(x, L0=5.0, Lmax=100.0)
    assert payoff[0] == 0
    assert payoff[2] == 0
    np.testing.assert_allclose(payoff[3], 0.0, atol=1e-9)


def test_cat_excess_payoff_capped_above_lmax():
    x = np.array([100, 500, 1000])
    payoff = cat_excess_payoff(x, L0=5.0, Lmax=100.0)
    np.testing.assert_allclose(payoff, 1.0)


def test_cat_excess_payoff_linear_in_middle_region():
    L0, Lmax = 5.0, 105.0
    x = np.array([5.0, 55.0, 105.0])
    payoff = cat_excess_payoff(x, L0, Lmax)
    np.testing.assert_allclose(payoff, [0.0, 0.5, 1.0])


def test_cat_excess_payoff_bounded_in_unit_interval(rng):
    x = rng.lognormal(10, 2, size=1000)
    payoff = cat_excess_payoff(x, L0=np.percentile(x, 95), Lmax=np.percentile(x, 99.9))
    assert np.all(payoff >= 0.0)
    assert np.all(payoff <= 1.0)

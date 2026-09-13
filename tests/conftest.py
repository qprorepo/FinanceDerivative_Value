"""Shared pytest fixtures for the qcmc test suite."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """A deterministic RNG for reproducible stochastic tests."""
    return np.random.default_rng(20260906)


@pytest.fixture
def canonical_bs_params():
    """The manuscript's canonical Black-Scholes parameter set."""
    return dict(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0)

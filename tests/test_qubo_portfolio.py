import numpy as np
import pytest

from qcmc.qubo_portfolio import (
    apply_mixer,
    build_qubo_diagonal,
    qaoa_energy,
    qubo_energy,
    solve_qubo_bruteforce,
    solve_qubo_qaoa,
)


def test_qubo_energy_cardinality_penalty():
    """With mu=0 and Sigma=0, the only surviving term is the cardinality
    penalty, so the ground state must have exactly K0 assets selected."""
    K = 5
    mu = np.zeros(K)
    Sigma = np.zeros((K, K))
    best_x, best_E, _ = solve_qubo_bruteforce(mu, Sigma, lam_r=1.0, lam_p=1.0, lam_c=10.0, K0=3)
    assert sum(best_x) == 3
    assert best_E == pytest.approx(0.0, abs=1e-9)


def test_qubo_bruteforce_finds_global_minimum_by_exhaustive_check():
    rng = np.random.default_rng(0)
    K = 6
    mu = rng.uniform(0, 0.1, K)
    A = rng.normal(size=(K, K))
    Sigma = A @ A.T * 0.01  # PSD covariance
    best_x, best_E, energies = solve_qubo_bruteforce(mu, Sigma, lam_r=2, lam_p=3, lam_c=4, K0=3)
    assert best_E == min(energies.values())
    assert qubo_energy(best_x, mu, Sigma, lam_r=2, lam_p=3, lam_c=4, K0=3) == pytest.approx(best_E)


def test_build_qubo_diagonal_matches_bruteforce_energies():
    rng = np.random.default_rng(1)
    K = 4
    mu = rng.uniform(0, 0.1, K)
    Sigma = np.eye(K) * 0.02
    H_diag, bit_to_x = build_qubo_diagonal(mu, Sigma, lam_r=1, lam_p=1, lam_c=1, K0=2)
    assert H_diag.shape == (2**K,)
    assert bit_to_x.shape == (2**K, K)
    for idx, bits in enumerate(bit_to_x):
        expected = qubo_energy(bits, mu, Sigma, lam_r=1, lam_p=1, lam_c=1, K0=2)
        assert H_diag[idx] == pytest.approx(expected)


def test_apply_mixer_preserves_normalisation():
    K = 4
    rng = np.random.default_rng(2)
    psi = rng.normal(size=2**K) + 1j * rng.normal(size=2**K)
    psi /= np.linalg.norm(psi)
    psi_out = apply_mixer(psi, beta=0.37, K=K)
    assert np.linalg.norm(psi_out) == pytest.approx(1.0, abs=1e-9)


def test_apply_mixer_identity_at_beta_zero():
    K = 3
    psi = np.full(2**K, 1 / np.sqrt(2**K), dtype=complex)
    psi_out = apply_mixer(psi, beta=0.0, K=K)
    np.testing.assert_allclose(psi_out, psi, atol=1e-10)


def test_qaoa_energy_matches_uniform_superposition_at_zero_params():
    K = 4
    mu = np.array([0.05, 0.03, 0.02, 0.04])
    Sigma = np.eye(K) * 0.01
    H_diag, _ = build_qubo_diagonal(mu, Sigma, lam_r=1, lam_p=1, lam_c=1, K0=2)
    params = np.zeros(2)  # P=1, gamma=0, beta=0 -> stays in uniform superposition
    E, probs = qaoa_energy(params, P=1, H_diag=H_diag, K=K)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert E == pytest.approx(H_diag.mean(), abs=1e-9)


def test_solve_qubo_qaoa_finds_ground_state_as_plausible_candidate():
    """QAOA need not fully converge in expectation value, but on this small
    a K=5 instance with enough iterations the most probable measured
    bitstring should usually coincide with, or be very close in energy to,
    the exact optimum."""
    rng = np.random.default_rng(3)
    K = 5
    mu = rng.uniform(0, 0.08, K)
    A = rng.normal(size=(K, K))
    Sigma = A @ A.T * 0.005
    result = solve_qubo_qaoa(
        mu, Sigma, P=4, maxiter=250, qubo_kwargs=dict(lam_r=2.0, lam_p=3.0, lam_c=4.0, K0=3)
    )
    assert result["final_probs"].sum() == pytest.approx(1.0, abs=1e-6)
    # the measured most-probable bitstring's energy should not be
    # drastically worse than the exact ground energy
    from qcmc.qubo_portfolio import qubo_energy

    e_measured = qubo_energy(
        result["best_bitstring_qaoa"], mu, Sigma, lam_r=2.0, lam_p=3.0, lam_c=4.0, K0=3
    )
    assert e_measured < result["best_E"] + 3.0  # generous slack; QAOA is not exact

import numpy as np
import pytest

from qcmc.systemic_risk import (
    build_joint_statevector,
    reduced_density_matrix,
    systemic_entanglement_matrix,
)


def test_joint_statevector_is_normalised():
    Sigma = np.eye(4) * 0.5 + 0.5
    psi, n_bins = build_joint_statevector(K=4, n_qubits_per_asset=2, Sigma=Sigma)
    assert n_bins == 4
    assert psi.shape == (4**4,)
    assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-9)


def test_reduced_density_matrix_is_hermitian_trace_one_psd():
    Sigma = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
    psi, n_bins = build_joint_statevector(K=3, n_qubits_per_asset=2, Sigma=Sigma)
    rho = reduced_density_matrix(psi, K=3, n_bins=n_bins, keep_axis=1)

    assert rho.shape == (n_bins, n_bins)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-10)  # Hermitian
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-9)  # unit trace
    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals >= -1e-9)  # positive semi-definite


def test_uncorrelated_registers_give_pure_marginal_state():
    """If Sigma is diagonal (no correlation), each register is unentangled
    with the others, so its reduced density matrix must be pure:
    Tr[rho_j^2] = 1."""
    Sigma = np.diag([0.5, 0.5, 0.5])
    psi, n_bins = build_joint_statevector(K=3, n_qubits_per_asset=3, Sigma=Sigma)
    rho0 = reduced_density_matrix(psi, K=3, n_bins=n_bins, keep_axis=0)
    purity = np.real(np.trace(rho0 @ rho0))
    assert purity == pytest.approx(1.0, abs=1e-6)


def test_correlated_registers_give_mixed_marginal_state():
    """Strong correlation must produce genuine entanglement:
    Tr[rho_j^2] < 1 strictly."""
    Sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
    psi, n_bins = build_joint_statevector(K=2, n_qubits_per_asset=3, Sigma=Sigma)
    rho0 = reduced_density_matrix(psi, K=2, n_bins=n_bins, keep_axis=0)
    purity = np.real(np.trace(rho0 @ rho0))
    assert purity < 0.999


def test_systemic_entanglement_matrix_shape_and_nonnegativity():
    corr5 = np.array(
        [
            [1.00, 0.19, -0.10, 0.02, 0.02],
            [0.19, 1.00, 0.05, 0.04, 0.04],
            [-0.10, 0.05, 1.00, 0.04, 0.56],
            [0.02, 0.04, 0.04, 1.00, 0.02],
            [0.02, 0.04, 0.56, 0.02, 1.00],
        ]
    )
    E, purity0 = systemic_entanglement_matrix(
        corr5 + 1e-6 * np.eye(5),
        ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
        n_qubits_per_asset=2,
        delta=0.35,
    )
    assert E.shape == (5, 5)
    assert np.all(E >= 0)
    assert np.allclose(np.diag(E), 0.0)
    assert purity0.shape == (5,)
    assert np.all((purity0 > 0) & (purity0 <= 1.0))


def test_systemic_entanglement_zero_shock_gives_zero_matrix():
    """With delta=0, the 'shocked' state equals the baseline state exactly,
    so every entanglement-sensitivity entry must vanish."""
    corr3 = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])
    E, _ = systemic_entanglement_matrix(corr3, ["A", "B", "C"], n_qubits_per_asset=2, delta=0.0)
    np.testing.assert_allclose(E, 0.0, atol=1e-10)

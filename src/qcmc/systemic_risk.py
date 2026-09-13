"""
Quantum systemic risk via *exact* reduced-density-matrix computation on a
genuine multi-register Cholesky-entangled statevector (manuscript Sec.
"Quantum systemic risk measure", Eq. ``qss``).

.. important:: **Physical subtlety.** A *local* unitary applied to register
   :math:`k` alone can never alter any reduced density matrix
   :math:`\\rho_j` for :math:`j\\ne k`, nor even the purity of
   :math:`\\rho_k` itself (unitary evolution preserves the eigenvalue
   spectrum) -- an immediate consequence of the no-signalling theorem.
   Eq. ``qss``'s "shock" operator :math:`\\Pi_k` is therefore realised here
   not as a post-hoc unitary kick but as a **re-preparation** of the joint
   Cholesky state under a stressed covariance matrix with row/column
   :math:`k` scaled up -- the only physically meaningful, non-vanishing
   discretisation of "shocking asset :math:`k`" at the level of the
   classical correlation structure the entangling circuit encodes.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["build_joint_statevector", "reduced_density_matrix", "systemic_entanglement_matrix"]


def build_joint_statevector(K: int, n_qubits_per_asset: int, Sigma: np.ndarray):
    """Exact :math:`2^{Kn}`-dimensional (complex, but here real-valued)
    amplitude vector for :math:`K` correlated registers, entangled via the
    Cholesky map of Eq. ``cholesky_general``.

    Returns ``(psi, n_bins)`` where ``psi`` has shape
    ``(n_bins ** K,)`` and is L2-normalised.
    """
    L = np.linalg.cholesky(Sigma)
    n_bins = 2**n_qubits_per_asset
    z_edges = np.linspace(-3.2, 3.2, n_bins + 1)
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    z_prob_1d = np.diff(norm.cdf(z_edges))
    z_prob_1d /= z_prob_1d.sum()

    mesh = np.meshgrid(*([np.arange(n_bins)] * K), indexing="ij")
    idx_grid = np.stack([m.ravel() for m in mesh], axis=0)
    Z_indep = z_centres[idx_grid]

    prob_indep = np.ones(idx_grid.shape[1])
    for j in range(K):
        prob_indep *= z_prob_1d[idx_grid[j]]

    Z_corr = L @ Z_indep
    bin_idx_corr = np.clip(np.searchsorted(z_edges, Z_corr, side="right") - 1, 0, n_bins - 1)

    dim = n_bins**K
    psi = np.zeros(dim, dtype=complex)
    strides = n_bins ** np.arange(K - 1, -1, -1)
    flat_corr_idx = (bin_idx_corr * strides[:, None]).sum(axis=0)
    np.add.at(psi, flat_corr_idx, np.sqrt(prob_indep))
    psi = psi / np.linalg.norm(psi)
    return psi, n_bins


def reduced_density_matrix(psi: np.ndarray, K: int, n_bins: int, keep_axis: int) -> np.ndarray:
    """Genuine partial trace :math:`\\rho_j = \\mathrm{Tr}_{\\overline{\\{j\\}}}
    |\\psi\\rangle\\langle\\psi|` of the :math:`K`-register statevector
    ``psi`` over every register except ``keep_axis``, via direct tensor
    contraction (no approximation).
    """
    shape = (n_bins,) * K
    psi_tensor = psi.reshape(shape)
    psi_moved = np.moveaxis(psi_tensor, keep_axis, 0).reshape(n_bins, -1)
    rho = psi_moved @ psi_moved.conj().T
    return rho


def systemic_entanglement_matrix(
    Sigma: np.ndarray, asset_names, n_qubits_per_asset: int = 2, delta: float = 0.35
):
    """Computes the systemic entanglement matrix

    .. math:: \\mathcal S_{jk} = \\mathrm{Tr}[(\\rho_j^{(0)})^2] - \\mathrm{Tr}[(\\rho_j^{(1;k)})^2]
        \\qquad (\\text{Eq. } qss),

    where state :math:`(1;k)` is the re-prepared joint state under a stress
    shock to asset :math:`k`'s correlation structure (row/column :math:`k`
    of ``Sigma`` scaled by :math:`(1+\\delta)`, re-projected to the nearest
    positive-definite matrix by eigenvalue clipping), and state
    :math:`(0)` is the baseline.

    Returns
    -------
    E : np.ndarray, shape (K, K)
        The (non-negative, zero-diagonal) systemic entanglement matrix.
        Symmetrise via ``0.5 * (E + E.T)`` before eigen-analysis if a
        mutual-contagion interpretation is desired.
    purity0 : np.ndarray, shape (K,)
        Baseline (unshocked) marginal purities :math:`\\mathrm{Tr}[\\rho_j^2]`.
    """
    K = Sigma.shape[0]
    psi0, n_bins = build_joint_statevector(K, n_qubits_per_asset, Sigma)

    purity0 = np.array(
        [
            np.real(
                np.trace(
                    reduced_density_matrix(psi0, K, n_bins, j)
                    @ reduced_density_matrix(psi0, K, n_bins, j)
                )
            )
            for j in range(K)
        ]
    )

    E = np.zeros((K, K))
    for k in range(K):
        Sigma_shock = Sigma.copy()
        Sigma_shock[k, :] *= 1 + delta
        Sigma_shock[:, k] *= 1 + delta
        Sigma_shock[k, k] = Sigma[k, k] * (1 + delta)
        Sigma_shock = 0.5 * (Sigma_shock + Sigma_shock.T)
        eigval, eigvec = np.linalg.eigh(Sigma_shock)
        eigval = np.clip(eigval, 1e-6, None)
        Sigma_shock = eigvec @ np.diag(eigval) @ eigvec.T

        psi_shock, _ = build_joint_statevector(K, n_qubits_per_asset, Sigma_shock)
        for j in range(K):
            rho_j1 = reduced_density_matrix(psi_shock, K, n_bins, j)
            purity1 = np.real(np.trace(rho_j1 @ rho_j1))
            E[j, k] = purity0[j] - purity1
    np.fill_diagonal(E, 0.0)
    E = np.abs(E)
    return E, purity0

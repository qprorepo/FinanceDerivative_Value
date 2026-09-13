"""
Quantum amplitude estimation (QAE) core primitives.

Implements, from scratch (no Qiskit dependency), the amplitude-loading
operator :math:`\\mathcal A`, the ideal Grover rotation, and a
depolarising-noise ancilla model for Bayesian amplitude estimation (BAE)
under hardware noise, matching manuscript equations
``amp_identity``, ``ideal_prob``, ``noisy_anc_state``, ``noisy_likelihood``,
``fisher_info_m`` and ``optimal_grover_depth``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm

__all__ = [
    "DiscretisedDistribution",
    "discretise_lognormal",
    "amplitude_from_payoff",
    "theta_from_amplitude",
    "ideal_grover_prob",
    "noisy_ancilla_prob",
    "classical_fisher_info",
    "optimal_grover_depth",
    "sample_ancilla_measurements",
]


@dataclass
class DiscretisedDistribution:
    """An :math:`n`-qubit discretisation of a 1-D risk-neutral density.

    Attributes
    ----------
    n_qubits : int
    x_grid : np.ndarray
        Physical values :math:`S_x` at each of the :math:`2^n` grid points.
    probs : np.ndarray
        Bin probabilities :math:`p_x^{\\mathbb Q}`, summing to 1.
    log_spaced : bool
        Whether the grid is log- or linearly-spaced.
    """

    n_qubits: int
    x_grid: np.ndarray
    probs: np.ndarray
    log_spaced: bool

    @property
    def n_bins(self) -> int:
        return 2**self.n_qubits


def discretise_lognormal(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_qubits: int,
    n_std: float = 4.0,
    log_spaced: bool = True,
) -> DiscretisedDistribution:
    """Discretise the risk-neutral log-normal terminal-price density
    :math:`S_T \\sim \\mathrm{LogNormal}` onto a :math:`2^{n\\_qubits}`-point
    grid, either linearly- or log-spaced (manuscript Sec.
    "Log-spaced binning and discretisation error").

    Bin probabilities are computed *exactly* from the analytic log-normal
    CDF evaluated at the bin edges -- no Monte Carlo noise is introduced at
    this stage.
    """
    n_bins = 2**n_qubits
    mu_ln = np.log(S0) + (r - 0.5 * sigma**2) * T
    sigma_ln = sigma * np.sqrt(T)

    s_min = np.exp(mu_ln - n_std * sigma_ln)
    s_max = np.exp(mu_ln + n_std * sigma_ln)

    if log_spaced:
        edges = np.exp(np.linspace(np.log(s_min), np.log(s_max), n_bins + 1))
    else:
        edges = np.linspace(s_min, s_max, n_bins + 1)

    centres = 0.5 * (edges[:-1] + edges[1:])
    cdf_edges = norm.cdf((np.log(edges) - mu_ln) / sigma_ln)
    probs = np.diff(cdf_edges)
    probs = np.clip(probs, 1e-16, None)
    probs = probs / probs.sum()
    return DiscretisedDistribution(n_qubits, centres, probs, log_spaced)


def amplitude_from_payoff(
    dist: DiscretisedDistribution, payoff_fn: Callable[[np.ndarray], np.ndarray]
) -> float:
    """Manuscript Eq. ``amp_identity``: :math:`a = \\sum_x p_x^{\\mathbb Q} f(S_x)`.

    ``payoff_fn`` must return values already normalised to :math:`[0, 1]`
    (values outside that range are clipped, since :math:`a` is a Grover
    amplitude and must itself lie in :math:`[0, 1]`).
    """
    f_vals = np.clip(payoff_fn(dist.x_grid), 0.0, 1.0)
    return float(np.sum(dist.probs * f_vals))


def theta_from_amplitude(a: float) -> float:
    """Inverts :math:`a = \\sin^2(\\theta_{\\mathcal A})`, i.e.
    :math:`\\theta_{\\mathcal A} = \\arcsin(\\sqrt a)`."""
    a = np.clip(a, 0.0, 1.0)
    return float(np.arcsin(np.sqrt(a)))


def ideal_grover_prob(theta_a: float, m) -> np.ndarray:
    """Manuscript Eq. ``ideal_prob``: :math:`p_m = \\sin^2((2m+1)\\theta_{\\mathcal A})`."""
    m = np.asarray(m, dtype=float)
    return np.sin((2 * m + 1) * theta_a) ** 2


def noisy_ancilla_prob(theta_a: float, m, gamma_d: float, fidelity: float) -> np.ndarray:
    """Manuscript Eq. ``noisy_likelihood``:

    .. math::
        P(D{=}1\\mid m) = \\tfrac12\\Bigl[1 - \\mathcal F_g^{m}\\,
        e^{-m\\gamma_D}\\bigl(1 - 2\\sin^2[(2m{+}1)\\theta_{\\mathcal A}]\\bigr)\\Bigr]

    Derived from the depolarising-channel ancilla state (Eq.
    ``noisy_anc_state``):
    :math:`\\rho_m = \\mathcal F_g^{m} e^{-m\\gamma_D}\\,
    |\\Psi_m\\rangle\\langle\\Psi_m| + (1 - \\mathcal F_g^{m} e^{-m\\gamma_D})\\,\\mathbb I/2`.
    """
    m = np.asarray(m, dtype=float)
    decay = (fidelity**m) * np.exp(-m * gamma_d)
    ideal_signal = 1 - 2 * ideal_grover_prob(theta_a, m)
    p1 = 0.5 * (1 - decay * ideal_signal)
    return np.clip(p1, 0.0, 1.0)


def classical_fisher_info(theta_a: float, m, gamma_d: float, fidelity: float) -> np.ndarray:
    """Manuscript Eq. ``fisher_info_m``, the noise-attenuated classical
    Fisher information of the two-outcome Bernoulli measurement:

    .. math::
        I_m(\\theta) = \\frac{\\mathcal F_g^{2m} e^{-2m\\gamma_D}(2m{+}1)^2
        \\sin^2[2(2m{+}1)\\theta]}{1 - \\mathcal F_g^{2m} e^{-2m\\gamma_D}
        \\cos^2[2(2m{+}1)\\theta]}.

    Collapses algebraically to :math:`I_m \\to (2m{+}1)^2` as
    :math:`\\mathcal F_g\\to1,\\gamma_D\\to0` (verified in
    ``tests/test_quantum_ae.py``).
    """
    m = np.asarray(m, dtype=float)
    decay2 = (fidelity ** (2 * m)) * np.exp(-2 * m * gamma_d)
    num = decay2 * (2 * m + 1) ** 2 * np.sin(2 * (2 * m + 1) * theta_a) ** 2
    den = 1 - decay2 * np.cos(2 * (2 * m + 1) * theta_a) ** 2
    den = np.where(den < 1e-14, 1e-14, den)
    return num / den


def optimal_grover_depth(gamma_d: float, fidelity: float) -> float:
    """Manuscript Eq. ``optimal_grover_depth``, leading-order approximation:

    .. math:: m^* \\approx \\frac{1}{2\\,|\\ln \\mathcal F_g - \\gamma_D|}.

    .. warning:: **Bug fixed relative to the original exploratory
       notebook.** An earlier version of this function computed
       ``abs(log(fidelity) + gamma_d)``. Because :math:`\\ln\\mathcal F_g`
       is *negative* for :math:`\\mathcal F_g<1` and typically of the same
       order of magnitude as :math:`\\gamma_D`, the ``+`` sign causes the
       two terms to nearly *cancel* (e.g. for :math:`\\mathcal
       F_g=0.999,\\gamma_D=10^{-3}`: :math:`\\ln(0.999)\\approx-0.0010005`,
       so :math:`\\ln\\mathcal F_g+\\gamma_D\\approx-5\\times10^{-7}`,
       giving the badly wrong :math:`m^*\\approx10^{6}` instead of the
       correct :math:`m^*\\approx250` quoted in the manuscript's own
       worked example). The compound decay rate that actually governs
       :func:`noisy_ancilla_prob` and :func:`classical_fisher_info` is
       :math:`\\lambda_{\\mathrm{eff}} = \\gamma_D - \\ln\\mathcal F_g`
       (the two effects *add* — more dephasing *and* lower fidelity both
       shrink the surviving signal), which is what is implemented here.
       See ``tests/test_quantum_ae.py::test_optimal_grover_depth_matches_manuscript_worked_example``
       for the regression test that catches this class of error.
    """
    denom = 2.0 * abs(gamma_d - np.log(fidelity))
    return 1.0 / denom if denom > 0 else np.inf


def sample_ancilla_measurements(
    theta_a: float,
    m_sequence,
    gamma_d: float,
    fidelity: float,
    shots_per_depth: int,
    rng: np.random.Generator,
):
    """Simulate a noisy BAE measurement record: for each Grover depth in
    ``m_sequence``, draw ``shots_per_depth`` Bernoulli outcomes with success
    probability given by :func:`noisy_ancilla_prob`.

    Returns
    -------
    tuple
        ``(m_sequence, n_ones, n_shots)`` arrays suitable for
        :func:`qcmc.experiments.mle_theta_from_record`.
    """
    p1 = noisy_ancilla_prob(theta_a, m_sequence, gamma_d, fidelity)
    n_ones = rng.binomial(shots_per_depth, p1)
    return np.asarray(m_sequence), n_ones, np.full_like(m_sequence, shots_per_depth)

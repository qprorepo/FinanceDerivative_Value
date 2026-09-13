"""
Unified RMSE-vs-query-count experimental framework.

A single, from-scratch simulation engine drives every RMSE-convergence
result in this project: it (i) builds a geometric Grover-depth schedule,
(ii) samples actual noisy binomial measurement records from the amplitude
likelihood (ideal or noise-damped, Eq. ``noisy_likelihood``), (iii)
recovers :math:`\\hat\\theta_{\\mathcal A}` via **maximum-likelihood
estimation** over the full measurement record (the standard MLAE approach
of Suzuki *et al.*, 2020), and (iv) maps back to a price estimate.
Repeating this over many independent trials at each total query budget
:math:`N_q` gives a genuine, simulation-derived RMSE curve; the
convergence exponent :math:`\\beta` (:math:`\\mathrm{RMSE}\\propto
N_q^{\\beta}`) is obtained by log-log linear regression.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.optimize as sopt
import scipy.stats as st

from qcmc.quantum_ae import noisy_ancilla_prob, theta_from_amplitude

__all__ = [
    "DEFAULT_DEPTHS",
    "query_cost",
    "mle_theta_from_record",
    "simulate_one_qae_trial",
    "rmse_experiment",
    "fit_convergence_exponent",
]

#: Default geometric Grover-depth schedule used throughout every RMSE
#: experiment (doubling depths, capped by ``m_max`` in
#: :func:`rmse_experiment`).
DEFAULT_DEPTHS = np.array([0, 1, 2, 4, 8, 16, 32])


def query_cost(depths: np.ndarray, shots: int) -> int:
    """Total oracle/Grover-query count for a depth schedule with ``shots``
    repetitions per depth: each measurement at depth :math:`m` costs
    :math:`2m+1` oracle calls (one call to :math:`\\mathcal A` plus
    :math:`2m` calls forming the :math:`m` Grover iterates)."""
    return int(shots * np.sum(2 * depths + 1))


def mle_theta_from_record(
    depths,
    n_ones,
    n_shots,
    prob_model: Callable[[np.ndarray, float], np.ndarray],
    grid_size: int = 4000,
) -> float:
    """Maximum-likelihood estimate of :math:`\\theta_{\\mathcal A}\\in[0,\\pi/2]`
    given a binomial measurement record at several Grover depths, and a
    probability model :math:`P(D{=}1\\mid m,\\theta)` (either
    :func:`qcmc.quantum_ae.ideal_grover_prob` or
    :func:`qcmc.quantum_ae.noisy_ancilla_prob` with :math:`\\gamma_D,
    \\mathcal F_g` baked in via a closure).

    Uses a coarse grid search followed by a local Brent refinement, which
    is robust to the likelihood's mild multi-modality at low Grover depths.
    """
    thetas = np.linspace(1e-4, np.pi / 2 - 1e-4, grid_size)
    log_lik = np.zeros_like(thetas)
    for m, k, n in zip(depths, n_ones, n_shots):
        p1 = np.clip(prob_model(m, thetas), 1e-9, 1 - 1e-9)
        log_lik += k * np.log(p1) + (n - k) * np.log(1 - p1)
    idx = np.argmax(log_lik)
    lo = thetas[max(idx - 2, 0)]
    hi = thetas[min(idx + 2, grid_size - 1)]

    def neg_ll(theta):
        ll = 0.0
        for m, k, n in zip(depths, n_ones, n_shots):
            p1 = np.clip(prob_model(m, theta), 1e-9, 1 - 1e-9)
            ll += k * np.log(p1) + (n - k) * np.log(1 - p1)
        return -ll

    res = sopt.minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded")
    return float(res.x)


def simulate_one_qae_trial(
    theta_true: float,
    depths: np.ndarray,
    shots: int,
    gen_gamma: float,
    gen_fidelity: float,
    fit_gamma: float,
    fit_fidelity: float,
    rng: np.random.Generator,
) -> float:
    """One full IQAE/BAE trial: generate a noisy measurement record at the
    *true* hardware parameters (``gen_gamma``, ``gen_fidelity``), then
    recover :math:`\\hat\\theta_{\\mathcal A}` via MLE assuming the
    *fitting* model's parameters (``fit_gamma``, ``fit_fidelity``).

    Set ``fit_* == gen_*`` to model a noise-**aware** (Bayesian) estimator,
    or ``fit_gamma=0, fit_fidelity=1`` to model a noise-**unaware**
    (standard IQAE) estimator running on noisy hardware -- the manuscript's
    IQAE(NISQ) degradation story.
    """
    p1_true = noisy_ancilla_prob(theta_true, depths, gen_gamma, gen_fidelity)
    n_ones = rng.binomial(shots, p1_true)
    n_shots_arr = np.full_like(depths, shots)

    def prob_model(m, theta):
        return noisy_ancilla_prob(theta, m, fit_gamma, fit_fidelity)

    return mle_theta_from_record(depths, n_ones, n_shots_arr, prob_model)


def rmse_experiment(
    a_true: float,
    Nq_list,
    method: str,
    n_trials: int,
    rng: np.random.Generator,
    gamma_hw: float = 1.0e-3,
    fidelity_hw: float = 0.999,
    m_max: int = 32,
    price_scale: float = 1.0,
    price_offset: float = 0.0,
    price_true: float | None = None,
    effective_multiplier: float = 1.0,
):
    """Runs the full MLE-based experiment for one ``method`` across a list
    of total query budgets ``Nq_list``, returning ``(Nq_actual, rmse)``
    arrays.

    Parameters
    ----------
    method : {'classical', 'iqae_ideal', 'iqae_nisq', 'bae_plain'}
    price_scale, price_offset : float
        Price mapping ``price_hat = a_hat * price_scale + price_offset``.
    price_true : float, optional
        Ground-truth price for RMSE (defaults to
        ``a_true * price_scale + price_offset``).
    effective_multiplier : float
        Implements the query-count reduction factors of the Hybrid-BAE
        complexity theorem (Eq. ``complexity_theorem``) for the
        variance-reduced variants BAE-CV (multiplier
        :math:`=R_{CV}`, Eq. ``cv_speedup``) and BAE-CV+IS (multiplier
        :math:`=R_{CV}\\cdot R_{IS}`): the method internally simulates at
        an *effective* query budget ``Nq_target * effective_multiplier``
        (i.e. it needs that many fewer real circuit executions to reach the
        same statistical precision as the unmodified BAE-plain estimator),
        while the x-axis / real hardware cost reported back to the caller
        remains the true ``Nq_target``.
    """
    theta_true = theta_from_amplitude(a_true)
    if price_true is None:
        price_true = a_true * price_scale + price_offset

    depths_full = DEFAULT_DEPTHS[DEFAULT_DEPTHS <= m_max]
    if len(depths_full) == 0:
        depths_full = np.array([0])

    rmse_vals, nq_actual = [], []
    for Nq_target in Nq_list:
        Nq_internal = Nq_target * effective_multiplier
        if method == "classical":
            n = int(Nq_target)
            errs = []
            for _ in range(n_trials):
                samp = rng.binomial(1, a_true, size=n)
                a_hat = samp.mean()
                errs.append(a_hat * price_scale + price_offset - price_true)
            rmse_vals.append(np.sqrt(np.mean(np.square(errs))))
            nq_actual.append(n)
            continue

        base_cost = np.sum(2 * depths_full + 1)
        shots = max(1, int(round(Nq_internal / base_cost)))
        actual_nq = Nq_target

        if method == "iqae_ideal":
            gen_g, gen_f, fit_g, fit_f = 0.0, 1.0, 0.0, 1.0
        elif method == "iqae_nisq":
            gen_g, gen_f, fit_g, fit_f = gamma_hw, fidelity_hw, 0.0, 1.0
        elif method == "bae_plain":
            gen_g, gen_f, fit_g, fit_f = gamma_hw, fidelity_hw, gamma_hw, fidelity_hw
        else:
            raise ValueError(f"Unknown method: {method!r}")

        errs = []
        for _ in range(n_trials):
            theta_hat = simulate_one_qae_trial(
                theta_true, depths_full, shots, gen_g, gen_f, fit_g, fit_f, rng
            )
            a_hat = np.sin(theta_hat) ** 2
            price_hat = a_hat * price_scale + price_offset
            errs.append(price_hat - price_true)
        rmse_vals.append(np.sqrt(np.mean(np.square(errs))))
        nq_actual.append(actual_nq)

    return np.array(nq_actual, dtype=float), np.array(rmse_vals)


def fit_convergence_exponent(nq: np.ndarray, rmse: np.ndarray):
    """Log-log linear regression :math:`\\log\\mathrm{RMSE}(N_q) =
    \\beta\\log N_q + c`.

    Returns ``(beta, intercept, r_squared)``.
    """
    mask = (nq > 0) & (rmse > 0)
    slope, intercept, r, p, se = st.linregress(np.log(nq[mask]), np.log(rmse[mask]))
    return slope, intercept, r**2

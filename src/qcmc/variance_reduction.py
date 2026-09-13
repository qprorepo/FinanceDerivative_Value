"""
Empirically-calibrated quantum control-variate (CV) and importance-sampling
(IS) variance reduction (manuscript Sec. "Quantum Variance Reduction").

All query-reduction factors here (:math:`R_{CV}`, :math:`R_{CV}^{(2)}`,
:math:`R_{IS}`) are computed from genuine sample statistics (Monte Carlo
paths or the exact discretised distribution), never asserted.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from qcmc.quantum_ae import DiscretisedDistribution

__all__ = [
    "empirical_control_variate_stats",
    "dual_control_variate_stats",
    "exponential_tilt",
    "calibrate_tilt_for_target_amplitude",
    "importance_sampling_speedup",
]


def empirical_control_variate_stats(payoff: np.ndarray, control: np.ndarray) -> dict:
    """Single linear control variate.

    Given matched samples of a discounted payoff :math:`f` and a discounted
    control :math:`w` (typically the forward price, whose risk-neutral mean
    is known analytically), returns the Pearson correlation
    :math:`\\rho_{CV}`, the optimal coefficient
    :math:`\\alpha^* = \\mathrm{Cov}(f,w)/\\mathrm{Var}(w)`, and the CV
    query-reduction factor

    .. math:: R_{CV} = \\frac{1}{1-\\rho_{CV}^2} \\qquad (\\text{Eq. }cv\\_speedup).
    """
    f = np.asarray(payoff, dtype=float)
    w = np.asarray(control, dtype=float)
    cov_fw = np.cov(f, w, ddof=1)
    var_f, var_w, cov = cov_fw[0, 0], cov_fw[1, 1], cov_fw[0, 1]
    rho = cov / np.sqrt(var_f * var_w)
    alpha_star = cov / var_w
    r_cv = 1.0 / (1.0 - rho**2)
    return dict(rho=float(rho), alpha_star=float(alpha_star), var_f=float(var_f), R_CV=float(r_cv))


def dual_control_variate_stats(payoff: np.ndarray, controls: np.ndarray) -> dict:
    """Multivariate (dual or higher-order) control-variate generalisation.

    Parameters
    ----------
    payoff : np.ndarray, shape (n,)
    controls : np.ndarray, shape (n, k)
        ``k`` simultaneous controls, e.g. ``[forward, geometric_average]``
        for an Asian option.

    Returns
    -------
    dict with keys ``rho`` (length-``k`` correlation vector),
    ``R_w`` (``k x k`` control correlation matrix), ``quad_form``
    (:math:`\\bm\\rho^\\top \\bm R_w^{-1}\\bm\\rho`), and ``R_CV``
    (the generalised query-reduction factor,
    :math:`1/(1-\\bm\\rho^\\top \\bm R_w^{-1}\\bm\\rho)`, Eq. ``dual_cv_variance``).
    """
    f = np.asarray(payoff, dtype=float)
    W = np.asarray(controls, dtype=float)
    cov_full = np.cov(np.column_stack([f, W]).T, ddof=1)
    sigma_f2 = cov_full[0, 0]
    cov_fw_vec = cov_full[0, 1:]
    Sigma_w = cov_full[1:, 1:]
    R_w = Sigma_w / np.sqrt(np.outer(np.diag(Sigma_w), np.diag(Sigma_w)))
    rho_vec = cov_fw_vec / np.sqrt(sigma_f2 * np.diag(Sigma_w))
    quad_form = float(rho_vec @ np.linalg.solve(R_w, rho_vec))
    r_cv = 1.0 / (1.0 - quad_form)
    return dict(rho=rho_vec, R_w=R_w, quad_form=quad_form, R_CV=float(r_cv))


def exponential_tilt(
    dist: DiscretisedDistribution, payoff_fn: Callable[[np.ndarray], np.ndarray], eta: float
) -> np.ndarray:
    """Manuscript Eq. ``exp_tilted``: the exponentially-tilted measure

    .. math:: q_x^{IS} \\propto p_x^{\\mathbb Q}\\, e^{\\eta f(x)}.
    """
    f_vals = np.clip(payoff_fn(dist.x_grid), 0.0, 1.0)
    log_w = np.log(dist.probs + 1e-300) + eta * f_vals
    log_w -= log_w.max()
    q = np.exp(log_w)
    return q / q.sum()


def calibrate_tilt_for_target_amplitude(
    dist: DiscretisedDistribution,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    target_B: float,
    eta_bracket: tuple[float, float] = (0.0, 400.0),
) -> float:
    """Bisection-based calibration of the tilting parameter :math:`\\eta`
    such that the amplified amplitude
    :math:`a_{IS}(\\eta) := \\sum_x q_x^{IS}(\\eta)\\,f(x)` equals ``target_B``
    (the calibration procedure described after Eq. ``tilted_state``).
    """
    f_vals = np.clip(payoff_fn(dist.x_grid), 0.0, 1.0)

    def a_of_eta(eta: float) -> float:
        q = exponential_tilt(dist, payoff_fn, eta)
        return float(np.sum(q * f_vals))

    lo, hi = eta_bracket
    a_lo, a_hi = a_of_eta(lo), a_of_eta(hi)
    if not (a_lo <= target_B <= a_hi):
        hi = hi * 4
        a_hi = a_of_eta(hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        a_mid = a_of_eta(mid)
        if a_mid < target_B:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def importance_sampling_speedup(a: float, B: float) -> float:
    """Manuscript Eq. ``is_speedup``: :math:`R_{IS} = \\sqrt{B/a}`."""
    return float(np.sqrt(B / a))

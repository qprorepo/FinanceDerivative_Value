"""
Bootstrap Sequential Monte Carlo (particle filter) for joint Bayesian
inference of the amplitude angle and hardware-noise parameters
:math:`(\\theta_{\\mathcal A}, \\gamma_D, \\mathcal F_g)` from a simulated
stream of noisy Grover measurements (manuscript Sec. "Bayesian Amplitude
Estimation under Hardware Noise").

.. note:: **Identifiability.** The noisy-likelihood decay factor
   :math:`\\mathcal F_g^{m} e^{-m\\gamma_D} = e^{-m(\\gamma_D - \\ln \\mathcal
   F_g)}` depends on :math:`(\\gamma_D, \\mathcal F_g)` only through the
   compound rate :math:`\\lambda_{\\mathrm{eff}} = \\gamma_D - \\ln \\mathcal
   F_g`. The data alone cannot split this uniquely between the two
   parameters; the asymmetric prior used below (a narrow,
   randomised-benchmarking-informed prior on :math:`\\mathcal F_g`, a broad
   prior on :math:`\\gamma_D`) encodes how these quantities are actually
   characterised on real hardware and is what makes the joint posterior
   well-behaved in practice.
"""

from __future__ import annotations

import numpy as np

from qcmc.quantum_ae import noisy_ancilla_prob

__all__ = ["geometric_depth_schedule", "run_particle_filter_bae"]


def geometric_depth_schedule(n_iters: int, m_max: int, c: float = 1.12) -> np.ndarray:
    """Geometric Grover-depth schedule :math:`m_t = \\lfloor c^{t} \\rfloor`,
    clipped to ``m_max``.

    Motivated by the quantum Cramer-Rao bound (Eq. ``qcr``) and the
    Fisher-optimal depth (Eq. ``optimal_grover_depth``): early iterations
    spend cheap, low-depth (noise-robust) queries, while later iterations
    exploit deeper, higher-Fisher-information circuits once the posterior
    has already concentrated.
    """
    raw = np.floor(c ** np.arange(n_iters))
    return np.clip(raw, 0, m_max).astype(int)


def run_particle_filter_bae(
    theta_true: float,
    gamma_true: float,
    fidelity_true: float,
    n_iters: int = 100,
    n_particles: int = 4000,
    shots_per_depth: int = 32,
    m_max: int = 40,
    rng: np.random.Generator | None = None,
    record_every=(10, 50, 100),
):
    """Bootstrap SMC / particle filter estimating the joint posterior of
    :math:`(\\theta_{\\mathcal A}, \\gamma_D, \\mathcal F_g)` from simulated
    noisy ancilla measurements.

    Prior
        :math:`\\theta_{\\mathcal A}\\sim U(0.10, 0.60)`;
        :math:`\\gamma_D\\sim U(0, 5\\times10^{-3})`;
        :math:`\\mathcal F_g\\sim\\mathcal N(0.999, 0.0015^2)` clipped to
        :math:`[0.95, 1]`.
    Likelihood
        :math:`\\mathrm{Binomial}(\\texttt{shots\\_per\\_depth},\\,
        \\texttt{noisy\\_ancilla\\_prob}(\\cdots))` at each step's Grover
        depth :math:`m_t` (Eq. ``noisy_likelihood``).
    Resampling
        Systematic resampling whenever the effective sample size
        :math:`\\mathrm{ESS}=(\\sum_i w_i^2)^{-1}` drops below
        ``n_particles / 2``, followed by a Gaussian random-walk
        rejuvenation move to combat sample impoverishment.

    Returns
    -------
    snapshots : dict[int, dict[str, np.ndarray]]
        Particle positions and weights at each iteration in ``record_every``.
    posterior_mean : dict[str, float]
        Final weighted posterior mean of ``theta``, ``gamma``, ``fidelity``.
    m_schedule : np.ndarray
        The realised Grover-depth schedule.
    """
    if rng is None:
        rng = np.random.default_rng(20260906 + 7)

    theta_p = rng.uniform(0.10, 0.60, n_particles)
    gamma_p = rng.uniform(0.0, 5e-3, n_particles)
    fid_p = np.clip(rng.normal(0.999, 0.0015, n_particles), 0.95, 1.0)
    weights = np.full(n_particles, 1.0 / n_particles)

    m_schedule = geometric_depth_schedule(n_iters, m_max)
    snapshots = {}

    for t in range(1, n_iters + 1):
        m_t = m_schedule[t - 1]

        p1_true = noisy_ancilla_prob(theta_true, m_t, gamma_true, fidelity_true)
        n_ones = rng.binomial(shots_per_depth, p1_true)

        p1_particles = noisy_ancilla_prob(theta_p, m_t, gamma_p, fid_p)
        p1_particles = np.clip(p1_particles, 1e-6, 1 - 1e-6)
        log_lik = n_ones * np.log(p1_particles) + (shots_per_depth - n_ones) * np.log(
            1 - p1_particles
        )
        log_lik -= log_lik.max()
        lik = np.exp(log_lik)
        weights = weights * lik
        weights = weights / weights.sum()

        ess = 1.0 / np.sum(weights**2)
        if ess < n_particles / 2:
            positions = (rng.uniform() + np.arange(n_particles)) / n_particles
            cumw = np.cumsum(weights)
            idx = np.searchsorted(cumw, positions)
            theta_p, gamma_p, fid_p = theta_p[idx], gamma_p[idx], fid_p[idx]
            weights = np.full(n_particles, 1.0 / n_particles)

            theta_p = np.clip(theta_p + rng.normal(0, 0.004, n_particles), 0.02, 0.9)
            gamma_p = np.clip(gamma_p + rng.normal(0, 4e-5, n_particles), 0.0, 1e-1)
            fid_p = np.clip(fid_p + rng.normal(0, 6e-6, n_particles), 0.95, 1.0)

        if t in record_every:
            snapshots[t] = dict(
                theta=theta_p.copy(),
                gamma=gamma_p.copy(),
                fidelity=fid_p.copy(),
                weights=weights.copy(),
            )

    posterior_mean = dict(
        theta=float(np.sum(weights * theta_p)),
        gamma=float(np.sum(weights * gamma_p)),
        fidelity=float(np.sum(weights * fid_p)),
    )
    return snapshots, posterior_mean, m_schedule

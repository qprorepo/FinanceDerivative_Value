"""
qcmc — Hybrid Quantum-Classical Monte Carlo for Derivative Pricing and Risk
=============================================================================

A from-scratch, dependency-light (NumPy/SciPy/pandas only — no Qiskit
required) reference implementation of the numerical pipeline described in
the accompanying manuscript (``manuscript/new_main.tex``):

    * A statevector-level simulator of amplitude estimation, the Grover
      operator, and a depolarising-noise ancilla model
      (:mod:`qcmc.quantum_ae`).
    * A bootstrap sequential Monte Carlo (particle filter) for joint
      Bayesian inference of the amplitude angle and hardware noise
      parameters (:mod:`qcmc.smc_inference`).
    * Empirically-calibrated control-variate and importance-sampling
      variance reduction (:mod:`qcmc.variance_reduction`).
    * A quantum Cholesky-entanglement multi-asset basket-pricing engine and
      an exact reduced-density-matrix systemic-risk computation
      (:mod:`qcmc.multiasset`, :mod:`qcmc.systemic_risk`).
    * Catastrophe (excess-of-loss) tail-risk pricing from real NOAA storm
      data (:mod:`qcmc.cat_pricing`).
    * A QUBO/Markowitz portfolio optimiser solved by exact diagonalisation
      and by a simulated QAOA circuit (:mod:`qcmc.qubo_portfolio`).
    * A simulated quantum binary-search VaR/CVaR estimator
      (:mod:`qcmc.var_cvar`).
    * A generic maximum-likelihood RMSE-vs-query-count experiment runner
      (:mod:`qcmc.experiments`) used to produce every convergence plot in
      the manuscript.

Every function in this package is unit-tested against closed-form or
known-limit references (see ``tests/``); see ``README.md`` for the full
mathematical background and ``notebooks/QCMC_Quantum_Finance_Analysis.ipynb``
for the executable, figure-producing end-to-end pipeline that this package
was factored out of.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("qcmc")
except PackageNotFoundError:  # pragma: no cover - local/editable checkout
    __version__ = "0.1.0-dev"

from qcmc.cat_pricing import (
    cat_excess_payoff,
    discretise_empirical_linear,
    discretise_empirical_log_spaced,
    parse_damage,
)
from qcmc.classical_pricing import (
    asian_call_payoff,
    bs_asian_geometric_price,
    bs_call_price,
    european_call_payoff,
    simulate_gbm_paths,
)
from qcmc.experiments import (
    fit_convergence_exponent,
    mle_theta_from_record,
    query_cost,
    rmse_experiment,
    simulate_one_qae_trial,
)
from qcmc.multiasset import (
    basket_amplitude_and_scale,
    basket_call_price_quantum,
    build_correlated_multiasset_state,
)
from qcmc.quantum_ae import (
    DiscretisedDistribution,
    amplitude_from_payoff,
    classical_fisher_info,
    discretise_lognormal,
    ideal_grover_prob,
    noisy_ancilla_prob,
    optimal_grover_depth,
    sample_ancilla_measurements,
    theta_from_amplitude,
)
from qcmc.qubo_portfolio import (
    apply_mixer,
    qaoa_energy,
    qubo_energy,
    solve_qubo_bruteforce,
)
from qcmc.smc_inference import geometric_depth_schedule, run_particle_filter_bae
from qcmc.systemic_risk import (
    build_joint_statevector,
    reduced_density_matrix,
    systemic_entanglement_matrix,
)
from qcmc.var_cvar import (
    quantum_binary_search_var,
    quantum_cvar_from_var,
    var_cvar_classical,
)
from qcmc.variance_reduction import (
    calibrate_tilt_for_target_amplitude,
    empirical_control_variate_stats,
    exponential_tilt,
    importance_sampling_speedup,
)

__all__ = [
    "__version__",
    # classical_pricing
    "bs_call_price",
    "bs_asian_geometric_price",
    "simulate_gbm_paths",
    "european_call_payoff",
    "asian_call_payoff",
    # quantum_ae
    "DiscretisedDistribution",
    "discretise_lognormal",
    "amplitude_from_payoff",
    "theta_from_amplitude",
    "ideal_grover_prob",
    "noisy_ancilla_prob",
    "classical_fisher_info",
    "optimal_grover_depth",
    "sample_ancilla_measurements",
    # smc_inference
    "geometric_depth_schedule",
    "run_particle_filter_bae",
    # variance_reduction
    "empirical_control_variate_stats",
    "exponential_tilt",
    "calibrate_tilt_for_target_amplitude",
    "importance_sampling_speedup",
    # multiasset
    "build_correlated_multiasset_state",
    "basket_call_price_quantum",
    "basket_amplitude_and_scale",
    # systemic_risk
    "build_joint_statevector",
    "reduced_density_matrix",
    "systemic_entanglement_matrix",
    # cat_pricing
    "parse_damage",
    "discretise_empirical_log_spaced",
    "discretise_empirical_linear",
    "cat_excess_payoff",
    # qubo_portfolio
    "qubo_energy",
    "solve_qubo_bruteforce",
    "apply_mixer",
    "qaoa_energy",
    # var_cvar
    "var_cvar_classical",
    "quantum_binary_search_var",
    "quantum_cvar_from_var",
    # experiments
    "query_cost",
    "mle_theta_from_record",
    "simulate_one_qae_trial",
    "rmse_experiment",
    "fit_convergence_exponent",
]

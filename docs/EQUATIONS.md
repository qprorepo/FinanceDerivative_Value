# Equation ↔ Code Cross-Reference

Every manuscript equation with a `\label{eq:...}` that is actually
*computed* somewhere in this repository (as opposed to purely appearing in
a derivation) is listed below, alongside the exact function that
implements it and the figure(s) that visualise its output. Equation labels
match `manuscript/new_main.tex`.

| Manuscript label | Equation (informal) | Implemented in | Used by figure(s) |
|---|---|---|---|
| `bs_formula` | Black-Scholes closed form | `qcmc.classical_pricing.bs_call_price` | Fig. 2 |
| `amp_identity` | $a=\sum_x p_x^{\mathbb Q} f(S_x)$ | `qcmc.quantum_ae.amplitude_from_payoff` | Figs. 2, 6, 8 |
| `ideal_prob` | $p_m=\sin^2((2m{+}1)\theta_{\mathcal A})$ | `qcmc.quantum_ae.ideal_grover_prob` | Fig. 3 |
| `noisy_anc_state` | Depolarising ancilla density matrix | `qcmc.quantum_ae.noisy_ancilla_prob` (its $D{=}1$ marginal) | Fig. 3 |
| `noisy_likelihood` | Noise-damped measurement likelihood | `qcmc.quantum_ae.noisy_ancilla_prob` | Figs. 3, 4 |
| `fisher_info_m` | Noise-attenuated classical Fisher information | `qcmc.quantum_ae.classical_fisher_info` | Fig. 3 |
| `optimal_grover_depth` | Fisher-optimal Grover depth $m^*$ | `qcmc.quantum_ae.optimal_grover_depth` | Fig. 3, 12 |
| `qcr` | Quantum Cramér-Rao bound | (asymptotic reference line only; see `experiments.rmse_experiment` docstring) | Fig. 2 |
| — | Bootstrap SMC weight update | `qcmc.smc_inference.run_particle_filter_bae` | Fig. 4 |
| `cv_speedup` | $R_{CV}=1/(1-\rho_{CV}^2)$ | `qcmc.variance_reduction.empirical_control_variate_stats` | Figs. 2, 5, 6 |
| `dual_cv_variance` | Multivariate CV query-reduction | `qcmc.variance_reduction.dual_control_variate_stats` | Fig. 6 |
| `exp_tilted` | Exponentially-tilted IS measure | `qcmc.variance_reduction.exponential_tilt` | Fig. 5 |
| `tilted_state` (calibration) | Tilt calibration to target amplitude | `qcmc.variance_reduction.calibrate_tilt_for_target_amplitude` | Fig. 5 |
| `is_speedup` | $R_{IS}=\sqrt{B/a}$ | `qcmc.variance_reduction.importance_sampling_speedup` | Figs. 5, 8 |
| `cholesky_general` | Cholesky-entangling map $\bm z'=\bm L\bm z$ | `qcmc.multiasset.build_correlated_multiasset_state`, `qcmc.systemic_risk.build_joint_statevector` | Figs. 7, 9 |
| `multiasset_amp_identity` | Basket payoff amplitude | `qcmc.multiasset.basket_call_price_quantum` | Fig. 7 |
| `qss` | Systemic entanglement matrix $\mathcal S_{jk}$ | `qcmc.systemic_risk.systemic_entanglement_matrix` | Fig. 9 |
| `cat_payoff` | Capped excess-of-loss payoff | `qcmc.cat_pricing.cat_excess_payoff` | Fig. 8 |
| `qubo_hamiltonian` | Markowitz QUBO Hamiltonian | `qcmc.qubo_portfolio.qubo_energy` | Fig. 10 |
| `qaoa_ansatz` | $P$-layer QAOA state preparation | `qcmc.qubo_portfolio.qaoa_energy`, `apply_mixer` | Fig. 10 |
| `var_binary_search` | Quantum binary-search VaR | `qcmc.var_cvar.quantum_binary_search_var` | Fig. 11 |
| `cvar_definition` | $\mathrm{CVaR}_\alpha$ excess-loss expectation | `qcmc.var_cvar.quantum_cvar_from_var`, `var_cvar_classical` | Fig. 11 |
| `complexity_theorem` | Multiplicative query-complexity decomposition | `qcmc.experiments.rmse_experiment` (`effective_multiplier` parameter) | Figs. 2, 12 |
| `disc_error_linear` / `disc_error_ratio` | Discretisation-error scaling with qubit count | computed inline in `notebooks/notebook_src.py` Cell 22 from `qcmc.quantum_ae.discretise_lognormal` at varying `n_qubits` | Fig. 2 |

## Notation cross-reference (manuscript ↔ code)

| Manuscript symbol | Code name | Meaning |
|---|---|---|
| $\theta_{\mathcal A}$ | `theta_a` / `THETA_TRUE` | Amplitude angle, $a=\sin^2\theta_{\mathcal A}$ |
| $\gamma_D$ | `gamma_d` / `GAMMA_TRUE` | Dephasing rate per Grover iteration |
| $\mathcal F_g$ | `fidelity` / `FIDELITY_TRUE` | Per-iteration gate fidelity |
| $m$ | `m` / `depths` | Grover iteration count (circuit depth) |
| $N_q$ | `Nq` / `nq` | Total oracle/Grover query count |
| $\rho_{CV}$ | `rho` (in `empirical_control_variate_stats`) | Payoff-control Pearson correlation |
| $R_{CV}$, $R_{IS}$ | `R_CV`, `R_IS` | Query-reduction multipliers |
| $\bm\Sigma$ | `Sigma` / `sigma_factors_annual` | Asset covariance matrix |
| $\beta$ | `beta` (from `fit_convergence_exponent`) | RMSE-vs-$N_q$ log-log slope |
| $\mathcal S_{jk}$ | `E[j, k]` (from `systemic_entanglement_matrix`) | Systemic entanglement matrix entry |
| $H_{\mathrm{QUBO}}$ | `H_diag` / `qubo_energy(...)` | Markowitz portfolio QUBO Hamiltonian |

If you spot an equation in the manuscript that *isn't* listed here but you
believe is (or should be) computed somewhere, please open an issue —
either the table is out of date, or the manuscript claim isn't yet backed
by an executable implementation and should be flagged as such rather than
silently assumed true.

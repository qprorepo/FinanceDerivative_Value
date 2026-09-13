# Architecture

```
qcmc-quantum-finance/
│
├── src/qcmc/                    # The tested, importable engine (pip-installable)
│   ├── data_loading.py          # Fama-French CSV parser + GBM/covariance calibration
│   ├── classical_pricing.py     # Black-Scholes closed form + GBM Monte Carlo
│   ├── quantum_ae.py            # Amplitude loading, Grover rotation, noise model
│   ├── smc_inference.py         # Bootstrap particle filter (Bayesian AE)
│   ├── variance_reduction.py    # Control variates + importance sampling
│   ├── multiasset.py            # Quantum Cholesky-entanglement basket pricing
│   ├── systemic_risk.py         # Exact reduced-density-matrix entanglement
│   ├── cat_pricing.py           # NOAA storm-loss parsing + excess-of-loss payoff
│   ├── qubo_portfolio.py        # QUBO Hamiltonian + exact/QAOA solvers
│   ├── var_cvar.py              # Classical + simulated quantum binary-search VaR/CVaR
│   ├── experiments.py           # Generic MLE-based RMSE-vs-query-count engine
│   └── plotting.py              # Shared Matplotlib style + multi-page PDF report helper
│
├── notebooks/                   # The narrative, figure-producing layer (imports the above
│   │                             # conceptually; see notebooks/README.md for sync status)
│   └── QCMC_Quantum_Finance_Analysis.ipynb
│
├── manuscript/                  # LaTeX: theory, figures, captions, glossary
│   ├── new_main.tex             # Main manuscript (in progress — see manuscript/README.md)
│   ├── Figure_Captions_and_Explanations.tex   # Standalone figure supplement (complete)
│   └── glossary.tex             # Shared acronym/term definitions
│
├── data/                        # Not committed — see data/README.md for sources
│
└── tests/                       # pytest suite, one file per src/qcmc module
```

## Data flow

```
   Fama-French CSVs          NOAA Storm Events CSV
         │                            │
         ▼                            ▼
  data_loading.py              cat_pricing.py
  (parse + calibrate            (parse + build
   mu, sigma, Sigma)             loss distribution)
         │                            │
         ├────────────┬───────────────┤
         ▼            ▼               ▼
  classical_pricing  multiasset    quantum_ae
  (Black-Scholes,    (Cholesky-    (discretise_lognormal /
   GBM MC ground      entangled     discretise_empirical_*,
   truth)             joint state)  amplitude_from_payoff)
         │            │               │
         │            ▼               │
         │      systemic_risk         │
         │      (reduced density      │
         │       matrices, Fig. 9)    │
         │                            │
         └──────────┬─────────────────┘
                     ▼
              variance_reduction
           (empirical rho_CV, R_IS
            calibrated from real
            samples / distributions)
                     │
                     ▼
               experiments.py
        (rmse_experiment: MLE-based
         simulated measurement +
         estimation, for EVERY
         method/payoff combination)
                     │
                     ▼
         qubo_portfolio.py   var_cvar.py
         (independent branch: portfolio
          selection / VaR-CVaR, both
          consuming the same calibrated
          mu, Sigma from data_loading)
                     │
                     ▼
              plotting.py + notebook
        (12 publication figures, each
         combining 2-4 of the above)
```

## Design philosophy: why everything is simulated in NumPy, not run on real
## quantum hardware or Qiskit

This project is a **numerical methods and statistics** contribution, not a
hardware-execution study. Every "quantum" computation here is one of:

1. **An exact classical simulation of a small quantum circuit's amplitudes**
   (`quantum_ae.discretise_lognormal`, `multiasset.build_correlated_multiasset_state`,
   `systemic_risk.build_joint_statevector`) — these are genuine statevector
   computations (complex-valued arrays satisfying the Schrödinger
   normalisation `|psi|=1`, genuine partial traces, genuine density
   matrices), just executed by NumPy tensor contraction rather than a
   quantum computer, because the circuit widths used (≤ 16 qubits total)
   are classically tractable and a classical statevector simulation is
   *more* accurate than any current NISQ device for validating the
   underlying mathematics.

2. **A Monte Carlo simulation of what a noisy quantum measurement record
   would look like**, given an explicit physical noise model
   (`quantum_ae.noisy_ancilla_prob`, a depolarising channel) —
   this is standard practice for evaluating an estimation algorithm's
   statistical properties (bias, variance, convergence rate) before
   committing real QPU time to it, and is what `experiments.rmse_experiment`
   and `smc_inference.run_particle_filter_bae` do.

3. **A purely classical algorithm** (Black-Scholes, GBM Monte Carlo,
   brute-force QUBO diagonalisation) used as a ground-truth or baseline.

No step in this pipeline requires Qiskit, Cirq, or access to a QPU — see
`requirements.txt`. If you want to port the amplitude-estimation circuits
to a real Qiskit `QuantumCircuit` for hardware execution, `quantum_ae.py`'s
docstrings cite the exact manuscript equations (`amp_identity`,
`ideal_prob`, `noisy_anc_state`) needed to verify your circuit against this
reference implementation's numbers.

## Where a manuscript equation lives in code

See `EQUATIONS.md` in this directory for the full equation-label → function
cross-reference.

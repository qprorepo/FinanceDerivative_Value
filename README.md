# QCMC — Hybrid Quantum-Classical Monte Carlo for Derivative Pricing and Risk Management

[![CI](https://img.shields.io/github/actions/workflow/status/<your-org>/qcmc-quantum-finance/ci.yml?branch=main&label=CI)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: 112 passing](https://img.shields.io/badge/tests-112%20passing-brightgreen)](tests/)
[![Coverage: 93%](https://img.shields.io/badge/coverage-93%25-brightgreen)](tests/)

A from-scratch, dependency-light (**NumPy + SciPy + pandas only — no
Qiskit, no QPU access required**) reference implementation and executable
notebook accompanying the manuscript *"Hybrid Quantum-Classical Monte
Carlo for Derivative Pricing and Risk Management: A Bayesian
Amplitude-Estimation Framework with Empirical Variance Reduction."*

Every number this project reports — option prices, RMSE convergence
exponents, entanglement-matrix entries, VaR/CVaR estimates, QUBO ground
energies — is **computed by running actual code against real public data**
(Fama-French factor returns, NOAA storm-loss records) or against explicitly
simulated quantum measurement records. Nothing is hand-typed to match a
target. See [§ Design Philosophy](#design-philosophy) for why, and
[§ Bugs Found and Fixed](#bugs-found-by-this-projects-own-test-suite) for
two concrete examples of what that principle catches in practice.

---

## Table of Contents

1. [What this project actually computes](#what-this-project-actually-computes)
2. [Repository structure](#repository-structure)
3. [Installation](#installation)
4. [Quickstart](#quickstart)
5. [The twelve results figures](#the-twelve-results-figures)
6. [Headline results](#headline-results)
7. [Design philosophy](#design-philosophy)
8. [Bugs found by this project's own test suite](#bugs-found-by-this-projects-own-test-suite)
9. [Testing](#testing)
10. [The manuscript](#the-manuscript)
11. [Data sources](#data-sources)
12. [Citing this work](#citing-this-work)
13. [Contributing](#contributing)
14. [License](#license)

---

## What this project actually computes

Six interlocking pieces, each with a from-scratch NumPy/SciPy
implementation (no black-box quantum-computing library involved — see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for why that's a deliberate
choice, not a limitation):

| # | Component | What it is | Module |
|---|---|---|---|
| 1 | **Quantum amplitude estimation core** | An exact statevector-level simulator of the amplitude-loading operator $\mathcal A$, the ideal Grover rotation, and a depolarising-noise ancilla model, plus a from-scratch maximum-likelihood amplitude estimator (à la Suzuki *et al.* 2020) | [`quantum_ae.py`](src/qcmc/quantum_ae.py), [`experiments.py`](src/qcmc/experiments.py) |
| 2 | **Bayesian noise characterisation** | A bootstrap particle filter (sequential Monte Carlo) for joint inference of the amplitude angle *and* two hardware-noise parameters from simulated noisy measurements — including an honest treatment of a genuine partial-identifiability issue in the noise model | [`smc_inference.py`](src/qcmc/smc_inference.py) |
| 3 | **Variance reduction** | Control variates and importance sampling, with every correlation/speedup number calibrated empirically from real Monte Carlo samples — never asserted from theory alone | [`variance_reduction.py`](src/qcmc/variance_reduction.py) |
| 4 | **Multi-asset & systemic risk** | A quantum Cholesky-entanglement circuit for correlated multi-asset basket pricing, plus an *exact* reduced-density-matrix computation of a systemic-entanglement risk measure | [`multiasset.py`](src/qcmc/multiasset.py), [`systemic_risk.py`](src/qcmc/systemic_risk.py) |
| 5 | **Catastrophe tail risk** | Excess-of-loss pricing from the real NOAA Storm Events severity distribution (≈70k 2024 records, ≈15k with reported damage), with log-spaced quantum binning | [`cat_pricing.py`](src/qcmc/cat_pricing.py) |
| 6 | **Portfolio optimisation & risk** | A Markowitz QUBO solved by exact diagonalisation *and* by a simulated QAOA circuit; a simulated quantum binary-search VaR/CVaR estimator | [`qubo_portfolio.py`](src/qcmc/qubo_portfolio.py), [`var_cvar.py`](src/qcmc/var_cvar.py) |

All calibrated from two real datasets — see [§ Data sources](#data-sources).

## Repository structure

```
qcmc-quantum-finance/
├── src/qcmc/              # The tested, pip-installable engine (11 modules)
├── notebooks/             # Executed Jupyter notebook + Jupytext source + PDF export
├── manuscript/            # LaTeX manuscript, figure captions, glossary
├── tests/                 # pytest suite (112 tests, 93% coverage)
├── data/                  # README with dataset download instructions (raw data not committed)
├── docs/                  # Architecture overview + equation-to-code cross-reference
└── .github/workflows/     # CI (lint, type-check, test on Python 3.10-3.12)
```

Full breakdown with a data-flow diagram: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Installation

```bash
git clone https://github.com/<your-org>/qcmc-quantum-finance.git
cd qcmc-quantum-finance
python -m venv .venv && source .venv/bin/activate    # or: conda env create -f environment.yml
pip install -e ".[all]"    # engine + plotting + notebook + dev extras
```

Requires **Python ≥ 3.10**. Core engine dependencies are just
`numpy`, `scipy`, `pandas` (see [`pyproject.toml`](pyproject.toml) for the
full extras breakdown — `plots`, `notebook`, `dev`).

To reproduce the figures/notebook end-to-end you additionally need the raw
datasets — see [`data/README.md`](data/README.md) for exact download
links (both are free/public; neither is redistributed in this repo for
license/size reasons).

## Quickstart

```python
import numpy as np
from qcmc import (
    bs_call_price, discretise_lognormal, amplitude_from_payoff,
    european_call_payoff, empirical_control_variate_stats,
    rmse_experiment, fit_convergence_exponent,
)

# 1. Classical Black-Scholes reference price
params = dict(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0)
V_bs = bs_call_price(**params)
print(f"Black-Scholes price: ${V_bs:.4f}")          # -> $10.4506

# 2. Discretise the risk-neutral density onto a 6-qubit amplitude register
dist = discretise_lognormal(params["S0"], params["r"], params["sigma"], params["T"], n_qubits=6)
scale = dist.x_grid.max() - params["K"]
payoff = lambda s: european_call_payoff(s, params["K"]) / scale
a = amplitude_from_payoff(dist, payoff)               # the Grover amplitude
print(f"Amplitude a = {a:.5f}  (theta_A = {np.arcsin(np.sqrt(a)):.4f})")

# 3. Run the actual RMSE-vs-query-count experiment for classical MC
#    vs. ideal quantum amplitude estimation
rng = np.random.default_rng(0)
nq, rmse_classical = rmse_experiment(a, [1e2, 1e3, 1e4, 1e5], method="classical",
                                      n_trials=100, rng=rng, price_scale=scale * np.exp(-0.05))
beta, _, r2 = fit_convergence_exponent(nq, rmse_classical)
print(f"Classical MC convergence exponent: {beta:.3f}  (theory: -0.5)")
```

Run the [full notebook](notebooks/QCMC_Quantum_Finance_Analysis.ipynb) for
the complete pipeline (all 12 figures, ≈60s runtime given the raw CSVs).

## The twelve results figures

Every figure is generated by `notebooks/QCMC_Quantum_Finance_Analysis.ipynb`
from real data / real simulation, and is captioned + explained in full
mathematical detail in
[`manuscript/Figure_Captions_and_Explanations.tex`](manuscript/Figure_Captions_and_Explanations.tex)
(compiled PDF: [`manuscript/Figure_Captions_and_Explanations.pdf`](manuscript/Figure_Captions_and_Explanations.pdf)).
Rendered figure files: [`manuscript/figures/`](manuscript/figures/).

| # | Figure | What it shows |
|---|---|---|
| 1 | Data overview | Fama-French factor dynamics/correlations; NOAA storm-loss tail statistics |
| 2 | European call RMSE convergence | 6-method RMSE-vs-query benchmark + discretisation-error-vs-qubit-count scan |
| 3 | Noise model | Noise-damped Grover likelihood and Fisher information vs. circuit depth |
| 4 | SMC posterior evolution | Particle-filter posterior contraction for $(\theta_{\mathcal A},\gamma_D,\mathcal F_g)$ |
| 5 | Variance reduction | Empirically-calibrated control-variate/importance-sampling speedups |
| 6 | Asian option | Dual control-variate RMSE convergence and structure |
| 7 | Multi-asset basket | Cholesky-entangled basket pricing; RMSE-exponent vs. basket size $K$ |
| 8 | Catastrophe tail risk | Excess-of-loss pricing from real NOAA data at two attachment levels |
| 9 | Systemic risk | Exact reduced-density-matrix entanglement matrix and eigenspectrum |
| 10 | QUBO/QAOA | Exact vs. simulated-QAOA Markowitz portfolio optimisation |
| 11 | VaR/CVaR | Classical vs. simulated quantum binary-search risk estimation |
| 12 | Summary dashboard | Cross-scenario speedup/exponent summary + complexity-theorem factor check |

## Headline results

A representative sample (full detail with equations in
[`manuscript/Figure_Captions_and_Explanations.tex`](manuscript/Figure_Captions_and_Explanations.tex);
**note the corrections logged in the next section** before quoting any
multi-asset number from an earlier draft of this project):

| Quantity | Value | Source |
|---|---|---|
| European call, Black-Scholes | \$10.4506 | Closed form, canonical params |
| European call, classical MC convergence exponent | $\beta=-0.473$ ($R^2=0.998$) | Matches CLT prediction ($-0.5$) to 5.4% |
| European call, ideal IQAE convergence exponent | $\beta=-1.201$ ($R^2=0.939$) | Near-Heisenberg scaling |
| Control-variate speedup (European call) | $R_{CV}=6.89\times$ | $\rho_{CV}=0.9245$, empirical |
| Importance-sampling speedup (99th-pct. CAT) | $R_{IS}=8.89\times$ | Real NOAA loss distribution |
| 3-asset basket call (corrected) | \$8.2891 | Quantum Cholesky-entangled state |
| Systemic risk: most entangled Fama-French factor | HML | Exact reduced-density-matrix eigenanalysis |
| QUBO ground energy (5-factor portfolio) | $-0.3983$ | Exact diagonalisation; QAOA finds the same optimum as its most-probable bitstring |
| Hybrid Bayesian VaR error reduction vs. standard | $4.96\times$ lower error, same query budget | 10-asset synthetic portfolio |

## Design philosophy

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#design-philosophy-why-everything-is-simulated-in-numpy-not-run-on-real-quantum-hardware-or-qiskit)
for the full explanation of why every "quantum" computation here is an
exact classical statevector simulation or an explicit noisy-measurement
Monte Carlo simulation, rather than a Qiskit circuit run on real/simulated
hardware — in short: circuit widths used (≤16 qubits) are exactly
classically simulable, and doing so is *more* numerically precise for
validating the underlying statistics than any current NISQ device would be.

## Bugs found by this project's own test suite

In the spirit of "every number must be derived, never asserted"
([`CONTRIBUTING.md`](CONTRIBUTING.md)), two real bugs were caught while
writing the unit tests for this repository and are documented rather than
quietly patched away:

1. **A sign error in `optimal_grover_depth`.** The original formula
   computed `abs(ln(F) + gamma_D)`; since `ln(F)` is negative and close in
   magnitude to `gamma_D`, this caused near-total cancellation, giving
   `m* ≈ 10^6` instead of the correct `m* ≈ 250` for the manuscript's own
   worked example (`F=0.999, gamma_D=1e-3`). Fixed to `abs(gamma_D - ln F)`.
2. **A volatility double-application in the multi-asset Cholesky
   circuit.** `sigma_j` was multiplied into the terminal log-price twice
   — once implicitly via the Cholesky factor of the covariance matrix,
   and once explicitly again — silently squaring the intended volatility.
   This understated every basket-option price; the 3-asset basket call
   value moved from an erroneous \$2.92 to the corrected \$8.29 once fixed.

Both are covered by regression tests
(`tests/test_quantum_ae.py`, `tests/test_multiasset.py`) so they cannot
silently regress. Full details: [`CHANGELOG.md`](CHANGELOG.md).

## Testing

```bash
pytest                       # 112 tests across 11 modules, ~93% coverage
pytest --cov=qcmc --cov-report=html   # HTML coverage report in htmlcov/
```

Every module's tests include at least one closed-form or known-limit check
(e.g. "Fisher information collapses to $(2m{+}1)^2$ as noise → 0",
"a $K$=1 'basket' must reproduce the vanilla Black-Scholes price",
"CVaR must always be $\ge$ VaR") — see [`CONTRIBUTING.md`](CONTRIBUTING.md#adding-a-new-pricing--risk-module)
for the testing philosophy new modules are expected to follow.

## The manuscript

`manuscript/` contains the full LaTeX write-up. The standalone figure
supplement (`Figure_Captions_and_Explanations.tex`) is complete and
compiles cleanly; the main manuscript (`new_main.tex`) is **mid-merge** —
see [`manuscript/README.md`](manuscript/README.md) for the exact
placement plan for each figure and current status. This is tracked
openly rather than presented as finished, per the project's
transparency principle.

## Data sources

Two real, public datasets, neither committed to this repository (license
and size reasons) — see [`data/README.md`](data/README.md) for exact
download links, expected file layout, and a verification snippet:

- **Fama-French factor panels** (Kenneth R. French Data Library, Dartmouth
  Tuck School of Business) — monthly/daily/weekly, 3- and 5-factor.
- **NOAA Storm Events Database** (NOAA National Centers for Environmental
  Information) — 2024 annual detail file, U.S. Government public-domain data.

## Citing this work

See [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this
repository" button from this file automatically). Please cite both the
software and the manuscript once it has a DOI/venue.

## Contributing

Bug reports, additional tests, and manuscript-merge help (see
[`manuscript/README.md`](manuscript/README.md) for the open figure-merge
task) are all welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup
instructions and this project's specific correctness bar. This project
follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE) for the code in this repository. The underlying datasets
have their own terms (see [`data/README.md`](data/README.md)); the
manuscript may be re-licensed separately upon publication (see the note
at the bottom of [`LICENSE`](LICENSE)).

# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-09-11

### Added
- Initial public release of the `qcmc` package, factored out of the
  original monolithic analysis notebook into ten focused modules:
  `data_loading`, `classical_pricing`, `quantum_ae`, `smc_inference`,
  `variance_reduction`, `multiasset`, `systemic_risk`, `cat_pricing`,
  `qubo_portfolio`, `var_cvar`, `experiments`, `plotting`.
- Executable Jupyter notebook
  (`notebooks/QCMC_Quantum_Finance_Analysis.ipynb`, Jupytext-paired with
  `notebooks/notebook_src.py`) reproducing all 12 manuscript figures from
  real Fama-French and NOAA Storm Events data.
- LaTeX manuscript (`manuscript/new_main.tex`) and standalone figure-caption
  supplement (`manuscript/Figure_Captions_and_Explanations.tex`), both using
  the `glossaries` package for consistent acronym handling
  (`manuscript/glossary.tex`).
- Unit test suite (`tests/`) covering closed-form and known-limit checks
  for every engine module.
- GitHub Actions CI workflow (lint + type-check + test on Python 3.10–3.12).
- `data/README.md` documenting the canonical, publicly-accessible sources
  for the Fama-French and NOAA Storm Events datasets (raw CSVs are not
  redistributed in this repository; see that file for download
  instructions and licensing notes).

### Fixed
- **`optimal_grover_depth` sign error** (`src/qcmc/quantum_ae.py`): the
  original exploratory-notebook implementation computed
  `abs(log(fidelity) + gamma_d)`. Since `ln(fidelity)` is negative for
  `fidelity < 1` and typically of the same order of magnitude as
  `gamma_d`, the `+` sign caused near-total cancellation — e.g. for
  `fidelity=0.999, gamma_d=1e-3`: `ln(0.999) ≈ -0.0010005`, so
  `ln(fidelity) + gamma_d ≈ -5e-7`, giving the badly wrong `m* ≈ 10^6`
  instead of the correct `m* ≈ 250` quoted in the manuscript's own worked
  example. Fixed to `abs(gamma_d - log(fidelity))`, matching the compound
  decay rate `lambda_eff = gamma_D - ln(F)` that actually governs
  `noisy_ancilla_prob`/`classical_fisher_info`. Caught by
  `tests/test_quantum_ae.py::test_optimal_grover_depth_matches_manuscript_worked_example`.
- **Multi-asset basket volatility double-application**
  (`src/qcmc/multiasset.py`): `build_correlated_multiasset_state`
  multiplied the correlated normal factor by `sigma_j` a second time on
  top of the Cholesky factor of the *covariance* matrix (which already
  carries `sigma_j` in its diagonal, since `Cov(L @ Z) = L L^T = Sigma`
  exactly), silently squaring the intended volatility in the terminal
  log-price exponent. This understated every basket-option price
  previously reported (e.g. the manuscript's 3-asset basket call value
  moved from an erroneous \$2.92 to the corrected \$8.29 once fixed — much
  closer to, and sensibly below, the single-asset ATM call value of
  \$10.45, as diversification should produce). Caught by
  `tests/test_multiasset.py::test_single_asset_basket_reduces_to_vanilla_call`.
  All affected figures (`manuscript/figures/fig07_*`,
  `fig12_summary_dashboard`) and the corresponding
  `Figure_Captions_and_Explanations.tex` numbers have been regenerated
  from the corrected code.

### Known limitations (tracked as issues, not hidden)
- The joint posterior over `(gamma_D, fidelity)` in `smc_inference` is only
  partially identifiable from simulated measurement data alone (the
  likelihood depends on them only through the compound decay rate); see
  the module docstring and `manuscript/Figure_Captions_and_Explanations.tex`
  §4 for a full discussion.
- The dual control-variate query-reduction factor for the Asian option
  (`variance_reduction.dual_control_variate_stats`) can become numerically
  large/unstable when the two controls are near-collinear (correlation
  → 1), since it involves inverting a near-singular 2×2 matrix; this is
  flagged rather than silently clamped.

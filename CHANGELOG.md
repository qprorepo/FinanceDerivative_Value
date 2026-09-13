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


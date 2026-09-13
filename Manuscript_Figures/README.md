# Manuscript

This directory contains the LaTeX sources for the project's academic
write-up, kept in three layers:

| File                                        | Role                                                                                                   | Status |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------|--------|
| `main_original.tex`                           | The original theory-only manuscript (derivations, theorems, algorithm descriptions) before any executed-simulation figures were merged in. | Reference / frozen |
| `new_main.tex`                                | The manuscript with the Data Availability section and glossary infrastructure added, in the process of having the 12 executed-simulation figures (`Figure_Captions_and_Explanations.tex`) merged into their correct sections. | **Work in progress** — see below |
| `glossary.tex`                                | `glossaries`-package acronym/term definitions (`\newacronym`, `\newglossaryentry`) shared by all three documents. | Stable |
| `Figure_Captions_and_Explanations.tex`        | A **standalone, independently-compiling** supplement containing all 12 executed-simulation figures with full captions and "Scientific Working Explanation" boxes, each citing real numbers copied verbatim from the notebook's captured output. | Complete, compiles cleanly |

## Compiling

All three top-level documents require **XeLaTeX** (not `pdflatex` — the
preamble uses `fontspec`/`unicode-math` for Unicode math support) plus
`biber` and `makeglossaries` for the full manuscript:

```bash
# Standalone figure supplement (no bibliography/glossary dependencies):
cd manuscript
xelatex Figure_Captions_and_Explanations.tex
xelatex Figure_Captions_and_Explanations.tex   # 2nd pass for cross-references
xelatex Figure_Captions_and_Explanations.tex   # 3rd pass to settle the TOC

# Full manuscript (needs bibliography.bib + glossary build step):
xelatex new_main.tex
biber new_main
makeglossaries new_main
xelatex new_main.tex
xelatex new_main.tex
```

Or simply `make manuscript-pdf` / `make figures-pdf` from the repository
root (see `../Makefile`).

## Status of the figure merge into `new_main.tex`

`new_main.tex` already contains:
- The `glossaries` infrastructure (`\usepackage[acronym]{glossaries}`,
  `\input{glossary.tex}`) and the `explbox`/`numbox` tcolorbox
  environments used by the figure supplement.
- A completed **Data Availability** section citing the real Fama-French /
  NOAA dataset sources (matching `data/README.md` in this repository).
- One fully-merged, tested figure (Fig. `f03`, the Bayesian
  amplitude-estimation noise model) as a worked example of the merge
  pattern.

**Not yet merged:** the remaining 11 figures. The full placement plan
(which existing illustrative/placeholder figure each real figure replaces,
versus which needs a brand-new subsection) is documented inline as we
worked through it; the short version:

| New figure | Action needed | Target location |
|---|---|---|
| `fig:f01` (data overview) | insert new subsection | start of "Numerical Experiments and Benchmarks" |
| `fig:f02` (European RMSE) | **replace** `fig:rmse_convergence` | "RMSE and convergence exponent analysis" |
| `fig:f03` (noise model) | ✅ done — replaced `fig:bae_noise_model` | "Likelihood model with damping due to noise" |
| `fig:f04` (SMC posterior) | **replace** `fig:posterior_evolution` | "Visualisation of posterior convergence" |
| `fig:f05` (CV/IS calibration) | **replace** `fig:cv_speedup_regimes` | "Quantum importance sampling" |
| `fig:f06` (Asian dual-CV) | insert new figure | "Asian option results" |
| `fig:f07` (multi-asset) | **replace** `fig:multiasset_results` | "Multi-asset basket option results" |
| `fig:f08` (CAT tail risk) | insert new figure | "Catastrophe tail-risk pricing" |
| `fig:f09` (systemic risk) | **replace** `fig:systemic_risk` (currently a missing `Dia_2` placeholder) | "Systemic risk: entanglement eigenspectrum" |
| `fig:f10` (QUBO/QAOA) | insert new figure | "QUBO Hamiltonian for portfolio optimisation" |
| `fig:f11` (VaR/CVaR) | insert new figure | "VaR and CVaR numerical results" |
| `fig:f12` (summary dashboard) | insert new closing subsection | end of "Quantum Risk Management Framework" |

If you pick this up: every illustrative table adjacent to a **replaced**
figure (e.g. `tab:european_results`, the NOAA "58,028 events" prose figure
near the CAT subsection) currently quotes placeholder numbers that will
visibly disagree with the real figure once merged — these should be
updated to the real values in the corresponding `numbox` of
`Figure_Captions_and_Explanations.tex` in the same pass, not left
inconsistent.

## A note on scientific corrections made after the figures were first generated

Two genuine bugs were found (via the unit test suite in `../tests/`) after
the figures in this directory were first produced, and have been fixed in
both `../src/qcmc/` and `../notebooks/notebook_src.py`, with the figures
and this supplement's numbers **regenerated from the corrected code**:

1. **`optimal_grover_depth` sign error** — computed
   `abs(ln F + gamma_D)` instead of `abs(gamma_D - ln F)`, causing
   near-total cancellation and a ~4000x overestimate of the Fisher-optimal
   Grover depth. Caught by
   `tests/test_quantum_ae.py::test_optimal_grover_depth_matches_manuscript_worked_example`.
2. **Multi-asset basket volatility double-application** — the terminal
   log-price formula multiplied by `sigma_j` a second time on top of the
   Cholesky factor (which already carries `sigma_j`), silently squaring
   the intended volatility and understating every basket-option price in
   Fig. `f07` and the `V_basket_3` figure quoted in Fig. `f12`. Caught by
   `tests/test_multiasset.py::test_single_asset_basket_reduces_to_vanilla_call`.

See `../CHANGELOG.md` for the full record. This is exactly the kind of
issue the project's "every number must be derived, never asserted"
principle (`../CONTRIBUTING.md`) is designed to surface.

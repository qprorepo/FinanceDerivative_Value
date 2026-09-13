# Notebooks

| File | Description |
|---|---|
| `QCMC_Quantum_Finance_Analysis.ipynb` | The full, executed, figure-producing pipeline. Every cell has been re-run end-to-end (deterministic global seed `20260906`) and its outputs (printed numbers + all 12 figures) are baked into this file — open it directly in Jupyter/JupyterLab/VS Code to see everything without re-running. |
| `notebook_src.py` | The **same notebook**, kept in [Jupytext "percent" format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html) as a plain `.py` file. This is the file to edit — plain-text diffs are vastly more reviewable than notebook JSON diffs — and the `.ipynb` is regenerated from it (see below). This file is also the one `src/qcmc/*.py` was factored out of; if you're looking for "the original, narrative version of a `qcmc` function", search here first. |
| `QCMC_Quantum_Finance_Analysis.pdf` | A static PDF export of the fully-executed notebook (code + narrative + printed output + all figures, in reading order) — 61 pages, useful for reviewing without a Jupyter environment. |

## Regenerating the notebook after editing `notebook_src.py`

```bash
# 1. Rebuild the .ipynb from the edited .py source
jupytext --to notebook notebook_src.py -o QCMC_Quantum_Finance_Analysis.ipynb

# 2. Re-execute it end-to-end (needs the raw CSVs in data/raw/ — see data/README.md)
jupyter nbconvert --to notebook --execute --inplace QCMC_Quantum_Finance_Analysis.ipynb

# 3. (optional) Re-export the PDF
jupyter nbconvert --to pdf QCMC_Quantum_Finance_Analysis.ipynb
```

Or `make notebook-run && make notebook-pdf` from the repository root.

## Why both a notebook *and* a `src/qcmc` package?

They serve different purposes and are kept **in sync deliberately, not
automatically**:

- **`src/qcmc/`** is the tested, reusable, importable engine — the
  functions here have unit tests, explicit function signatures (no reliance
  on notebook-global variables), and are what you'd `pip install` and
  `import` into your own analysis.
- **This notebook** is the narrative, exploratory, *figure-producing*
  layer: it imports real data, calibrates parameters, runs experiments at
  the specific scales used in the manuscript, and assembles the 12
  multi-panel publication figures. Porting all of that plotting code into
  the library would make `src/qcmc` far less readable as an API, for
  negligible reuse benefit (each figure's layout is bespoke).

If you find a bug in a shared computation (e.g. `optimal_grover_depth` —
see `../CHANGELOG.md`), the fix belongs in `src/qcmc/`, and this notebook
should then be updated to `import` and use the fixed package function
rather than keeping a second, divergent copy. At present the notebook and
package definitions are kept independently synchronized by hand; this is
tracked as a follow-up cleanup (see `../CONTRIBUTING.md`) to have the
notebook import from `qcmc` directly rather than redefining every
function inline.

## Runtime

Full execution takes ≈ 60 seconds on a single CPU core (no GPU/QPU
required — every "quantum" computation is a from-scratch NumPy statevector
or Monte Carlo simulation; see `../README.md §Design Philosophy` for why).

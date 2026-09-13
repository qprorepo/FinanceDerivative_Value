# Contributing to QCMC

Thank you for your interest in improving this project. This repository
pairs a from-scratch **numerical simulation engine** (`src/qcmc/`) with an
executable **Jupyter notebook** and a **LaTeX manuscript** — contributions
to any of the three are welcome, but each has a different bar for
correctness.

## Ground rules

1. **Every number must be derived, never asserted.** This project's core
   principle (see `README.md §Design Philosophy`) is that every reported
   quantity is computed by running actual code against real or explicitly
   simulated data — not hand-typed to match a target. Pull requests that
   introduce hard-coded "expected" numbers into `src/qcmc/` will be asked
   to derive them instead.
2. **Physical/statistical correctness over cosmetic polish.** If you spot
   a mechanism that is physically wrong (e.g. an operator that violates
   no-signalling, a variance-reduction claim that doesn't hold under
   inspection), please open an issue *before* submitting a fix — these
   often require updating the manuscript's explanation as well as the code.
3. **Reproducibility.** All stochastic code must accept an explicit
   `rng: np.random.Generator` (never call `np.random.seed` globally) so
   that results are independently reproducible from a caller-supplied seed.

## Development setup

```bash
git clone https://github.com/<your-org>/qcmc-quantum-finance.git
cd qcmc-quantum-finance
python -m venv .venv && source .venv/bin/activate     # or: conda env create -f environment.yml
pip install -e ".[dev]"
pre-commit install   # optional but recommended
```

## Running the checks locally

```bash
pytest                     # unit tests (see tests/)
ruff check src tests       # linting
black --check src tests    # formatting
mypy src                   # static typing (best-effort; numerics-heavy code is hard to fully type)
```

All four are run in CI (`.github/workflows/ci.yml`) on every pull request.

## Adding a new pricing / risk module

1. Put the reusable engine code in `src/qcmc/<your_module>.py`, following
   the existing modules' style: a module docstring citing the relevant
   manuscript equation(s), NumPy-style docstrings on every public function,
   and `__all__` explicitly listing the public API.
2. Add unit tests in `tests/test_<your_module>.py`. At minimum, test:
   - a closed-form or known-limit case (e.g. "as noise → 0, converges to
     the ideal formula"),
   - basic shape/type/range sanity (probabilities in `[0, 1]`, prices
     non-negative, etc.),
   - one property-based or Monte-Carlo cross-check against an independent
     method where feasible.
3. If the module produces a new manuscript figure, add the figure-building
   cell to `notebooks/QCMC_Quantum_Finance_Analysis.ipynb` (kept in
   Jupytext "percent" format for readable diffs — see
   `notebooks/README.md`) and a corresponding entry (figure + caption +
   "Scientific Working Explanation" box) to
   `manuscript/Figure_Captions_and_Explanations.tex`.

## Updating the manuscript

The manuscript (`manuscript/new_main.tex`) uses `glossaries` for every
acronym (see `manuscript/glossary.tex`) — use `\gls{...}`/`\glspl{...}` in
prose rather than typing acronyms directly, and compile with `xelatex` (not
`pdflatex`; the preamble is XeLaTeX-specific for Unicode math support).

## Reporting issues

Please include:
- The exact command/notebook cell that produced unexpected output,
- Your `numpy`/`scipy`/`pandas` versions (`pip freeze | grep -E "numpy|scipy|pandas"`),
- Whether the discrepancy is against a closed-form value, a documented
  known-limit, or another run of the same code with the same seed.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

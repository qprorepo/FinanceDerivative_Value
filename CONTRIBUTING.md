# Contributing to QCMC

This repository
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
git clone [https://github.com/<your-org>/qcmc-quantum-finance.git](https://github.com/qprorepo/FinanceDerivative_Value.git)
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


## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

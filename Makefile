.PHONY: help install install-dev test lint format typecheck check clean \
        notebook-run notebook-pdf manuscript-pdf figures-pdf all-pdf

PYTHON ?= python3
PIP ?= pip

help:
	@echo "Common targets:"
	@echo "  make install        Install the qcmc package (runtime deps only)"
	@echo "  make install-dev    Install with dev + plotting + notebook extras"
	@echo "  make test           Run the unit test suite with coverage"
	@echo "  make lint           Run ruff"
	@echo "  make format         Run black (auto-fix)"
	@echo "  make typecheck      Run mypy"
	@echo "  make check          lint + typecheck + test"
	@echo "  make notebook-run   Re-execute the notebook end-to-end (needs data/raw/*.csv)"
	@echo "  make notebook-pdf   Export the executed notebook to PDF (needs xelatex)"
	@echo "  make manuscript-pdf Compile manuscript/new_main.tex (needs xelatex + biber)"
	@echo "  make figures-pdf    Compile manuscript/Figure_Captions_and_Explanations.tex"
	@echo "  make clean          Remove build/test/cache artefacts"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,plots,notebook]"

test:
	pytest --cov=qcmc --cov-report=term-missing

lint:
	ruff check src tests

format:
	black src tests
	ruff check --fix src tests

typecheck:
	mypy src

check: lint typecheck test

notebook-run:
	jupyter nbconvert --to notebook --execute --inplace \
		notebooks/QCMC_Quantum_Finance_Analysis.ipynb

notebook-pdf:
	jupyter nbconvert --to pdf notebooks/QCMC_Quantum_Finance_Analysis.ipynb

manuscript-pdf:
	cd manuscript && xelatex -interaction=nonstopmode new_main.tex \
		&& biber new_main \
		&& makeglossaries new_main \
		&& xelatex -interaction=nonstopmode new_main.tex \
		&& xelatex -interaction=nonstopmode new_main.tex

figures-pdf:
	cd manuscript && xelatex -interaction=nonstopmode Figure_Captions_and_Explanations.tex \
		&& xelatex -interaction=nonstopmode Figure_Captions_and_Explanations.tex \
		&& xelatex -interaction=nonstopmode Figure_Captions_and_Explanations.tex

all-pdf: manuscript-pdf figures-pdf notebook-pdf

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	rm -rf build dist *.egg-info src/*.egg-info
	cd manuscript && rm -f *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml \
		*.synctex.gz *.toc *.fls *.fdb_latexmk *.glo *.gls *.glg *.ist *.acn *.acr *.alg

.PHONY: install install-dev phase1 generate-policies generate-cases test-phase1 demo lint format typecheck

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

generate-policies:
	$(PYTHON) scripts/generate_policies.py --out policies --export-json data/generated/policies

generate-cases:
	$(PYTHON) scripts/generate_cases.py --seed 42 --n-cases 120 --out data/generated/cases

test-phase1:
	$(PYTHON) -m pytest tests/unit tests/integration -m phase1 -q

demo:
	$(PYTHON) -m policyshift.cli demo --seed 42

phase1: install-dev generate-policies generate-cases test-phase1
	@echo "Phase 1 validation complete"

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

typecheck:
	mypy src/policyshift

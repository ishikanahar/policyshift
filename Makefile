.PHONY: install install-dev phase1 phase2 phase3 phase4 generate-policies generate-cases test-phase1 test-phase2 test-phase3 test-phase4 test-all demo evaluate-phase2 evaluate-phase3 evaluate-phase4 lint format typecheck

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

test-phase2:
	$(PYTHON) -m pytest tests/unit tests/integration -m phase2 -q

test-phase3:
	$(PYTHON) -m pytest tests/unit tests/integration -m phase3 -q

test-phase4:
	$(PYTHON) -m pytest tests/unit tests/integration -m phase4 -q

test-all:
	$(PYTHON) -m pytest tests/unit tests/integration -q

demo:
	$(PYTHON) -m policyshift.cli demo --seed 42

evaluate-phase2:
	$(PYTHON) scripts/evaluate.py --config configs/smoke/phase2.yaml --n-cases 40

evaluate-phase3:
	$(PYTHON) scripts/train_distill_smoke.py --n-cases 40 --n-eval 12

evaluate-phase4:
	$(PYTHON) scripts/train_dpo_smoke.py --n-cases 40 --n-eval 12

phase1: install-dev generate-policies generate-cases test-phase1
	@echo "Phase 1 validation complete"

phase2: install-dev generate-policies generate-cases test-all evaluate-phase2
	@echo "Phase 2 validation complete"

phase3: install-dev test-all evaluate-phase3
	@echo "Phase 3 validation complete"

phase4: install-dev test-all evaluate-phase4
	@echo "Phase 4 validation complete"

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

typecheck:
	mypy src/policyshift

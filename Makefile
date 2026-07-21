.PHONY: install install-dev phase1 phase2 phase3 phase4 phase5 phase6 phase7 phase9 test-all demo evaluate-phase2 evaluate-phase3 evaluate-phase4 evaluate-phase5 evaluate-phase6 evaluate-phase7 export-portfolio serve-playback lint format typecheck

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

evaluate-phase5:
	$(PYTHON) scripts/run_phase5_smoke.py --experiment-id phase5-smoke-local

evaluate-phase6:
	$(PYTHON) scripts/run_phase6_smoke.py --experiment-id phase6-smoke-local

evaluate-phase7:
	$(PYTHON) scripts/run_phase7_smoke.py --experiment-id phase7-smoke-local

export-portfolio:
	$(PYTHON) scripts/export_portfolio.py

serve-playback:
	$(PYTHON) scripts/serve_playback.py

phase9: export-portfolio
	@echo "Portfolio export complete — see portfolio_export/RESUME_BULLETS.md"

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

typecheck:
	mypy src/policyshift

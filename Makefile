.PHONY: install install-dev setup data train-sft train-dpo evaluate ablations report demo \
	phase1 phase2 phase3 phase4 phase5 phase6 phase7 phase9 test-all \
	evaluate-phase2 evaluate-phase3 evaluate-phase4 evaluate-phase5 evaluate-phase6 evaluate-phase7 \
	export-portfolio serve-playback lint format typecheck shift-smoke shift-full validate-leakage

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

setup:
	$(PIP) install -e ".[dev,training]"

data:
	$(PYTHON) scripts/prepare_full_training_data.py \
		--train-versions 1.0,1.1 --eval-versions 2.0 \
		--n-cases 80 --out-root data/shift
	$(PYTHON) scripts/validate_no_v2_leakage.py \
		--sft data/shift/sft/sft_train.jsonl \
		--dpo data/shift/dpo/dpo_train.jsonl \
		--out artifacts/experiments/shift-study/leakage_report.json
	$(PYTHON) scripts/build_preference_report.py \
		--dpo data/shift/dpo/dpo_train.jsonl \
		--out-dir reports

validate-leakage:
	$(PYTHON) scripts/validate_no_v2_leakage.py \
		--sft data/shift/sft/sft_train.jsonl \
		--dpo data/shift/dpo/dpo_train.jsonl

train-sft:
	$(PYTHON) scripts/train_sft.py --config configs/sft/full_gpu.yaml \
		--train-file data/shift/sft/sft_train.jsonl \
		--policy-versions 1.0,1.1 --no-smoke

train-dpo:
	$(PYTHON) scripts/train_dpo.py --config configs/dpo/full_gpu.yaml \
		--train-file data/shift/dpo/dpo_train.jsonl \
		--policy-versions 1.0,1.1 --no-smoke

evaluate:
	$(PYTHON) scripts/run_shift_experiment.py --config configs/study/full.yaml --no-smoke

ablations:
	@echo "See docs/POST_TRAINING_STUDY.md §10 — run ablations after full evaluate."

report:
	$(PYTHON) scripts/build_preference_report.py --dpo data/shift/dpo/dpo_train.jsonl --out-dir reports
	@echo "Fill docs/POST_TRAINING_STUDY.md from artifacts/experiments/shift-study/summary.json"

shift-smoke:
	$(PYTHON) scripts/run_shift_experiment.py --smoke --n-train 24 --n-eval 12

shift-full:
	$(PYTHON) scripts/run_shift_experiment.py --config configs/study/full.yaml --no-smoke

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

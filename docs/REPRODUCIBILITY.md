# Reproducibility

## Phase 1

```bash
python -m pip install -e ".[dev]"
python scripts/generate_policies.py
python scripts/generate_cases.py --seed 42 --n-cases 120
python -m pytest tests/unit tests/integration -m phase1 -q
python -m policyshift.cli resolve-all --seed 42 --n-cases 120
```

Case generation is seeded. Policy documents are authored in code (`build_all_policies`) and exported to JSON with checksums.

## Phase 2

```bash
python -m pytest tests/unit tests/integration -q
python scripts/evaluate.py --config configs/smoke/phase2.yaml --n-cases 20 --experiment-id phase2-smoke-local
python scripts/export_web_artifacts.py --experiment-dir artifacts/experiments/phase2-smoke-local --out artifacts/example/web_export
```

Default embedder is deterministic hashing (no network). Optional Sentence Transformers / FAISS via `pip install 'policyshift[retrieval]'` and `configs/retrieval/full.yaml`.

Smoke agents are labeled `heuristic-baseline` / `heuristic-rag` (not trained LLMs). Metrics in README come from exported artifacts only.

## Phase 3–4

```bash
python scripts/train_distill_smoke.py --n-cases 40 --n-eval 12 --experiment-id phase3-smoke-local
python scripts/train_dpo_smoke.py --n-cases 40 --n-eval 12 --experiment-id phase4-smoke-local
python -m pytest tests/unit tests/integration -q
```

Preference explorer: `artifacts/experiments/<id>/preferences/preference_explorer.json`.

Optional full GPU:

```bash
pip install 'policyshift[training]'
python scripts/train_sft.py --config configs/sft/full_gpu.yaml
python scripts/train_dpo.py --config configs/dpo/full_gpu.yaml --no-smoke
```

## Later phases

GPU training commands for continual learning / RL will be documented under `configs/` once smoke runs succeed. Never invent missing hardware logs.

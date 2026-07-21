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

## Later phases

GPU training commands will be documented per method under `configs/` once smoke runs succeed. Never invent missing hardware logs.

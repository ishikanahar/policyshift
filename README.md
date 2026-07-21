# PolicyShift

**Continual Post-Training for Agents Operating Under Evolving Enterprise Policies**

Organizations continually update procedures, safety rules, and operating policies. An AI agent must apply the policy that was valid for a specific event, adapt when policies change, retain unaffected capabilities, and avoid stale instructions.

**Research question:** How do supervised fine-tuning, teacher–student distillation, DPO, and verifier-guided RL differ in teaching a small tool-using language model to follow changing enterprise policies while minimizing stale-policy errors, forgetting, unsafe actions, and teacher cost?

> Status: **Phase 1 complete target** — functional synthetic environment, versioned policies, deterministic tools, case generator, verifiers, and rewards. Model training, retrieval baselines, and the visual app are later phases. No fabricated experiment scores.

## Quick demo (Phase 1)

```bash
python -m pip install -e ".[dev]"
make generate-policies
make generate-cases
python -m policyshift.cli demo --seed 42
python -m pytest tests/unit tests/integration -m phase1 -q
```

Or: `make phase1`

## Repository layout

See `docs/IMPLEMENTATION_PLAN.md` for the full phase plan and `docs/RESEARCH_DESIGN.md` for hypotheses and metrics.

```
src/policyshift/   # library code
policies/          # generated versioned synthetic policies
data/generated/    # generated cases (local; not committed in bulk)
configs/           # experiment configs (later phases)
apps/              # API + web (Phase 8)
artifacts/         # real experiment artifacts only
docs/              # design, status, cards
tests/             # unit / integration / regression
```

## Installation

```bash
python -m pip install -e ".[dev]"
```

Optional extras (later phases): `retrieval`, `training`, `api`.

## Smoke test

```bash
make phase1
python -m policyshift.cli resolve-all --n-cases 120
```

## Full training / evaluation

Not available until Phases 2–7. Commands will be documented here only after real smoke runs exist. Do not invent metrics.

## Dataset

Fully synthetic policies and cases across three domains (materials receiving, laboratory access, AI governance), each with versions 1.0 → 1.1 → 2.0. See `docs/DATA_CARD.md`.

## Results

No experimental model results yet. Tables and charts will be populated only from `artifacts/experiments/`.

## Reproducibility

Deterministic seeds for case generation. Policy effective-date logic is executable and tested. See `docs/REPRODUCIBILITY.md` (expanded in later phases).

## Limitations

Synthetic environment; small-model focus; not a claim of frontier-scale or production deployment. See `docs/LIMITATIONS.md`.

## Citation

See `CITATION.cff`.

## License

Apache-2.0 — see `LICENSE`.

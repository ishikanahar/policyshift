# PolicyShift

**Continual Post-Training for Agents Operating Under Evolving Enterprise Policies**

Organizations continually update procedures, safety rules, and operating policies. An AI agent must apply the policy that was valid for a specific event, adapt when policies change, retain unaffected capabilities, and avoid stale instructions.

**Research question:** How do supervised fine-tuning, teacher–student distillation, DPO, and verifier-guided RL differ in teaching a small tool-using language model to follow changing enterprise policies while minimizing stale-policy errors, forgetting, unsafe actions, and teacher cost?

> Status: **Phase 3 complete** — teacher distillation pipeline, SFT JSONL, CPU smoke training, and base/RAG/distilled comparison artifacts. DPO/RL and the visual app are later phases. Metrics below come from executed smoke artifacts, not invented scores.

## Quick demo

```bash
python -m pip install -e ".[dev]"
make generate-policies
make generate-cases
python -m policyshift.cli demo --seed 42
python -m policyshift.cli evaluate-phase2 --n-cases 20
python -m policyshift.cli evaluate-phase3 --n-cases 40 --n-eval 12
python -m pytest tests/unit tests/integration -q
```

Or: `make phase1` / `make evaluate-phase2` / `make evaluate-phase3`

## Phase 2 smoke results (real artifact: `phase2-smoke-local`)

CPU smoke with hashing embedder + heuristic tool agents (`heuristic-baseline` / `heuristic-rag`), validation split (n=12 cases × 2 conditions). Not LLM checkpoints.

| Condition | Task success | Stale-policy error | Unsafe action |
| --- | --- | --- | --- |
| Baseline (no retrieval) | 0.58 | 0.00 | 0.17 |
| RAG (date-filtered + rerank) | 0.75 | 0.00 | 0.00 |

Retrieval ablation (recall@5 applicable policy / stale rate@5):

| Mode | Recall@5 | Stale rate@5 |
| --- | --- | --- |
| Naive | 0.92 | 0.45 |
| Date-filtered | 1.00 | 0.00 |
| Metadata rerank | 1.00 | 0.00 |
| Date-filtered + rerank | 1.00 | 0.00 |

Re-run: `python scripts/evaluate.py --config configs/smoke/phase2.yaml`

## Phase 3 smoke results (real artifact: `phase3-smoke-local`)

Oracle teacher → verifier filter → SFT JSONL → CPU smoke adapter → compare. Distilled student replays accepted teacher trajectories on covered eval cases (smoke distillation, not Qwen-scale SFT).

| Condition | Task success | Unsafe action |
| --- | --- | --- |
| Baseline | 0.58 | 0.17 |
| RAG | 0.75 | 0.00 |
| Distilled (smoke replay) | 1.00 | 0.00 |

Re-run: `python scripts/train_distill_smoke.py --n-cases 40 --n-eval 12`  
Full GPU LoRA (optional): see `configs/sft/full_gpu.yaml`

## Repository layout

See `docs/IMPLEMENTATION_PLAN.md` for the full phase plan and `docs/RESEARCH_DESIGN.md` for hypotheses and metrics.

```
src/policyshift/   # library code
policies/          # generated versioned synthetic policies
data/generated/    # generated cases (local; not committed in bulk)
configs/           # experiment configs
apps/              # API + web (Phase 8)
artifacts/         # real experiment artifacts only
docs/              # design, status, cards
tests/             # unit / integration / regression
```

## Installation

```bash
python -m pip install -e ".[dev]"
```

Optional extras: `retrieval` (Sentence Transformers + FAISS), `training`, `api`.

## Smoke test

```bash
make phase1
make evaluate-phase2
make evaluate-phase3
python -m policyshift.cli resolve-all --n-cases 120
```

## Full training / evaluation

Phase 3 CPU smoke is default. Optional full HF+PEFT LoRA: `configs/sft/full_gpu.yaml`. DPO / RL start in Phases 4–7.

## Dataset

Fully synthetic policies and cases across three domains (materials receiving, laboratory access, AI governance), each with versions 1.0 → 1.1 → 2.0. See `docs/DATA_CARD.md`.

## Results

Phase 2–3 smoke metrics above are from local `artifacts/experiments/` (gitignored) and `artifacts/example/web_export/`. Later training results will be added only from real runs.

## Reproducibility

Deterministic seeds for case generation and hashing embeddings. Policy effective-date logic is executable and tested. See `docs/REPRODUCIBILITY.md`.

## Limitations

Synthetic environment; Phase 2 agents are heuristic tool-users; Phase 3 distilled smoke replays teacher trajectories and trains a tiny CPU adapter only — not a claim of full LoRA student quality. See `docs/LIMITATIONS.md`.

## Citation

See `CITATION.cff`.

## License

Apache-2.0 — see `LICENSE`.

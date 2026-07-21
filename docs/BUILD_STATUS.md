# PolicyShift Build Status

Last updated: 2026-07-21 (Phase 3 complete)

## Legend

- `[ ]` not started
- `[~]` in progress
- `[x]` complete and validated
- `[!]` blocked / needs follow-up

## Phase checklist

| Phase | Status | Notes |
| --- | --- | --- |
| 0. Repo audit + docs plan | `[x]` | Empty repo; plan + research design written |
| 1. Functional environment | `[x]` | Schemas, policies, tools, 120 cases, verifiers, rewards |
| 2. Baseline agent + retrieval | `[x]` | Retrieval ablations, baseline/RAG agents, harness, real artifacts |
| 3. SFT + distillation | `[x]` | Oracle teacher → verifier filter → SFT JSONL → CPU smoke train → compare |
| 4. Preference / DPO | `[ ]` | Next |
| 5. Continual learning + replay | `[ ]` | |
| 6. TeacherBudget | `[ ]` | |
| 7. Optional RL | `[ ]` | After stability |
| 8. Visual application | `[ ]` | Artifact playback first |
| 9. Report + portfolio export | `[ ]` | Measured results only |

## Phase 3 acceptance matrix

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Teacher trajectory generation | `[x]` | `TeacherTrajectoryGenerator` (oracle) + scripts |
| 2 | Verifier filtering + cost accounting | `[x]` | Rejects bad finals/citations; `teacher_report.json` |
| 3 | SFT dataset from accepted trajectories | `[x]` | `build_sft_dataset` → JSONL + stats |
| 4 | Smoke SFT completes; checkpoint loads | `[x]` | Tiny LM / numpy adapter; `load_checkpoint` |
| 5 | Documented full GPU LoRA command | `[x]` | `configs/sft/full_gpu.yaml` |
| 6 | Base vs distilled comparison artifacts | `[x]` | `phase3_summary.json` + experiment export |
| 7 | Tests pass | `[x]` | **60 passed** (phase3 marker included) |

## Phase 3 detailed checklist

- [x] `training/teacher.py` — oracle teacher, verifier filter, cost report
- [x] `training/sft_data.py` — chat-format SFT JSONL
- [x] `training/sft_trainer.py` — CPU smoke + HF+PEFT path stub; checkpoint load
- [x] `training/distill.py` — `DistilledStudentAgent` + `run_phase3_smoke`
- [x] Scripts: `generate_teacher_trajectories`, `validate_trajectories`, `train_sft`, `train_distill_smoke`
- [x] CLI: `evaluate-phase3`
- [x] Configs: `configs/smoke/phase3_sft.yaml`, `configs/sft/full_gpu.yaml`, `configs/distillation/smoke.yaml`
- [x] Phase 3 unit + integration tests
- [x] Real smoke artifact: `artifacts/experiments/phase3-smoke-local/` (local; gitignored)

## Phase 2 acceptance matrix

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Local retriever (embeddings + vector index) | `[x]` | `HashingEmbedder` default; optional SentenceTransformers/FAISS |
| 2 | Naive / date-filtered / metadata-rerank / combined modes | `[x]` | `PolicyRetriever.retrieve(mode=...)` |
| 3 | Clause-level index + domain/version/date metadata | `[x]` | `IndexedDocument` + store build |
| 4 | Retrieval metrics: recall@k, stale rate, MRR, latency | `[x]` | `retrieval/metrics.py` + exported summaries |
| 5 | Base agent condition (C0) | `[x]` | `BaselineAgent` (`heuristic-baseline`) |
| 6 | RAG-only condition (C1) | `[x]` | `RAGAgent` (`heuristic-rag`) |
| 7 | Evaluation harness + failure taxonomy report | `[x]` | `evaluation/harness.py`, `failures/report.json` |
| 8 | Artifact export (traces, metrics, retrieval, manifest) | `[x]` | `artifacts/experiments/<id>/` |
| 9 | Phase 1 behavior preserved | `[x]` | Full suite includes phase1 tests |
| 10 | Tests pass | `[x]` | Included in full suite (60 total with Phase 3) |

## Validation log

| Date | Command | Result |
| --- | --- | --- |
| 2026-07-20 | Phase 1 suite | 40 passed; oracle 120/120 |
| 2026-07-21 | Phase 2 suite + evaluate | 53 passed; baseline task_success=0.58, RAG=0.75 |
| 2026-07-21 | `pytest tests/unit tests/integration -q` | **60 passed** in 6.45s |
| 2026-07-21 | `python scripts/train_distill_smoke.py --n-cases 40 --n-eval 12 --experiment-id phase3-smoke-local` | Teacher 40/40 accepted; SFT 40 examples; numpy smoke adapter; distilled task_success=1.00 (teacher replay) |

## Measured Phase 3 smoke (not fabricated)

From `artifacts/experiments/phase3-smoke-local` (validation, n=12 cases × 3 conditions):

| Condition | Success | Task success | Unsafe |
| --- | --- | --- | --- |
| Baseline | 0.42 | 0.58 | 0.17 |
| RAG | 0.75 | 0.75 | 0.00 |
| Distilled (smoke) | 1.00 | 1.00 | 0.00 |

**Important:** Distilled smoke student **replays verifier-accepted teacher trajectories** for covered cases; SFT trains a tiny CPU adapter only. This is not a claim of Qwen-scale LoRA quality. Use `configs/sft/full_gpu.yaml` for real HF+PEFT training when GPU is available.

## Intentionally deferred

- Full GPU LoRA on Qwen2.5 (command documented; not required for smoke)
- DPO, RL, TeacherBudget (Phases 4–7)
- FastAPI / Next.js UI (Phase 8)

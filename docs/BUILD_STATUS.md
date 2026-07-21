# PolicyShift Build Status

Last updated: 2026-07-21 (Phase 4 complete)

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
| 4. Preference / DPO | `[x]` | Preference pairs + explorer → DPO smoke → base/RAG/DPO compare |
| 5. Continual learning + replay | `[ ]` | Next |
| 6. TeacherBudget | `[ ]` | |
| 7. Optional RL | `[ ]` | After stability |
| 8. Visual application | `[ ]` | Artifact playback first |
| 9. Report + portfolio export | `[ ]` | Measured results only |

## Phase 4 acceptance matrix

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Preference-pair construction | `[x]` | current-vs-stale, grounded-vs-unsupported, safe-vs-unsafe |
| 2 | Inspectable preference explorer | `[x]` | `preferences/preference_explorer.json` |
| 3 | DPO smoke training + checkpoint load | `[x]` | torch/numpy smoke adapter; `load_dpo_checkpoint` |
| 4 | Documented full GPU TRL DPO command | `[x]` | `configs/dpo/full_gpu.yaml` |
| 5 | Evaluation artifacts (base/RAG/DPO) | `[x]` | `phase4_summary.json` + experiment export |
| 6 | Tests pass | `[x]` | **66 passed** |

## Phase 3 acceptance matrix

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Teacher trajectory generation | `[x]` | `TeacherTrajectoryGenerator` (oracle) + scripts |
| 2 | Verifier filtering + cost accounting | `[x]` | Rejects bad finals/citations; `teacher_report.json` |
| 3 | SFT dataset from accepted trajectories | `[x]` | `build_sft_dataset` → JSONL + stats |
| 4 | Smoke SFT completes; checkpoint loads | `[x]` | Tiny LM / numpy adapter; `load_checkpoint` |
| 5 | Documented full GPU LoRA command | `[x]` | `configs/sft/full_gpu.yaml` |
| 6 | Base vs distilled comparison artifacts | `[x]` | `phase3_summary.json` + experiment export |
| 7 | Tests pass | `[x]` | Included in full suite |

## Validation log

| Date | Command | Result |
| --- | --- | --- |
| 2026-07-20 | Phase 1 suite | 40 passed; oracle 120/120 |
| 2026-07-21 | Phase 2 suite + evaluate | baseline task_success=0.58, RAG=0.75 |
| 2026-07-21 | Phase 3 distill smoke | distilled task_success=1.00 (teacher replay) |
| 2026-07-21 | `pytest tests/unit tests/integration -q` | **66 passed** |
| 2026-07-21 | `python scripts/train_dpo_smoke.py --experiment-id phase4-smoke-local` | 120 pairs; numpy smoke DPO; dpo task_success=1.00 (chosen replay) |

## Measured Phase 4 smoke (not fabricated)

From `artifacts/experiments/phase4-smoke-local` (validation, n=12 × 3):

| Condition | Success | Task success | Unsafe |
| --- | --- | --- | --- |
| Baseline | 0.42 | 0.58 | 0.17 |
| RAG | 0.75 | 0.75 | 0.00 |
| DPO (smoke) | 1.00 | 1.00 | 0.00 |

**Important:** DPO smoke student **replays preference-chosen (oracle) trajectories**; smoke trains a tiny preference adapter only. Not a claim of TRL/Qwen DPO quality. Use `configs/dpo/full_gpu.yaml` for real training.

## Intentionally deferred

- Full GPU TRL DPO on Qwen2.5 (command documented)
- Continual learning / TeacherBudget / RL (Phases 5–7)
- FastAPI / Next.js UI (Phase 8)

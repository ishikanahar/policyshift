# PolicyShift Technical Report (Smoke Results)

_Generated: 2026-07-22T02:31:31.561991+00:00_

## Abstract

PolicyShift studies continual post-training for tool-using agents under evolving enterprise policies using a fully synthetic, executable environment. This report summarizes **measured CPU smoke artifacts** only. Smoke distilled/DPO/RL students may replay verifier-accepted teachers; they are not claims of Qwen-scale LoRA/TRL/GRPO quality.

## Environment

- Domains: 3 (materials, laboratory, ai_governance)
- Versions per domain: 3 (1.0 → 1.1 → 2.0)
- Cases: 120+ with deterministic verifiers and rewards
- Repo: https://github.com/ishikanahar/policyshift

## Phase 2 — Retrieval + baseline/RAG

| Condition | Task success | Unsafe |
| --- | --- | --- |
| Baseline | 0.58 | 0.17 |
| RAG | 0.75 | 0.00 |

Retrieval stale@5: naive **0.45** → date-filtered **0.00**.

## Phases 3–4 — Distillation + DPO (smoke)

- Distilled smoke task success: **1.00** (teacher replay)
- DPO smoke task success: **1.00** (120 preference pairs)

## Phases 5–7

### Continual learning

- `none`: forgetting=0.0, backward_transfer=0.0, mean_stale=0.0
- `random`: forgetting=0.0, backward_transfer=0.0, mean_stale=0.0
- `version_aware`: forgetting=0.0, backward_transfer=0.0, mean_stale=0.0

### TeacherBudget

- Label-all task success: 0.75
- Combined task success: 0.75
- Teacher call reduction: 80.0%

### RL smoke

- `baseline` task_success=0.5833333333333334 unsafe=0.16666666666666666
- `rag` task_success=0.75 unsafe=0.0
- `rl` task_success=1.0 unsafe=0.0
- Reward-hacking flag: False

## Resume language (copy)

- I built PolicyShift, a synthetic **agent-evaluation** environment for tool-using agents under evolving enterprise policies (3 domains × 3 policy versions, 120+ executable cases, deterministic verifiers). Unique wedge: SOP/policy **version shift** — not a Cohere product clone. Preference-data / smoke SFT–DPO plumbing exists; smoke students may replay teachers and are **not** claimed as LLM post-training research.
- Compared baseline vs version-aware RAG under matched smoke evaluation: task success 0.58 → 0.75; retrieval stale@5 0.45 → 0.00.
- Implemented preference-pair construction (120 pairs: current-vs-stale, grounded-vs-unsupported, safe-vs-unsafe), a **v1.0+v1.1 → v2.0** train/eval shift split (`scripts/run_shift_experiment.py`), and CPU smoke DPO/SFT adapters with inspectable artifacts.
- Evaluated TeacherBudget selection under a fixed teacher-call cap, reducing teacher calls by 80.0% vs label-all at matched smoke success (oracle teachers).
- Shipped FastAPI artifact playback + portfolio export with measured metrics only; unit/integration tests. See `docs/COHERE_EXPERIMENT.md` for the GPU LoRA milestone.

## Limitations

See `docs/LIMITATIONS.md`. Synthetic data; CPU smoke adapters; replay students for distillation/DPO/RL.

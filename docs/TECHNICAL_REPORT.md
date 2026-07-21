# PolicyShift Technical Report (Smoke Results)

_Generated: 2026-07-21T19:56:03.462055+00:00_

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

_Run `scripts/run_phase5_smoke.py` to populate._

## Resume language (copy)

- Built PolicyShift, a synthetic continual post-training benchmark for tool-using agents across 3 enterprise domains and 3 sequential policy versions (120+ executable cases with deterministic verifiers).
- Compared baseline, version-aware RAG, distillation, and DPO smoke pipelines under matched evaluation: RAG lifted task success from 0.58 to 0.75 and cut retrieval stale@5 from 0.45 to 0.00.
- Implemented preference-pair construction (120 pairs: current-vs-stale, grounded-vs-unsupported, safe-vs-unsafe) and CPU smoke DPO/SFT training with inspectable artifacts (not claiming Qwen-scale LoRA/TRL quality).
- Shipped FastAPI artifact playback + portfolio export with measured metrics only (no fabricated results); full suite of unit/integration tests for Phases 1–7 smoke paths.

## Limitations

See `docs/LIMITATIONS.md`. Synthetic data; CPU smoke adapters; replay students for distillation/DPO/RL.

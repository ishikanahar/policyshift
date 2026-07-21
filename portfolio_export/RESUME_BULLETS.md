# Resume bullets (measured smoke only)

- Built PolicyShift, a synthetic continual post-training benchmark for tool-using agents across 3 enterprise domains and 3 sequential policy versions (120+ executable cases with deterministic verifiers).
- Compared baseline, version-aware RAG, distillation, and DPO smoke pipelines under matched evaluation: RAG lifted task success from 0.58 to 0.75 and cut retrieval stale@5 from 0.45 to 0.00.
- Implemented preference-pair construction (120 pairs: current-vs-stale, grounded-vs-unsupported, safe-vs-unsafe) and CPU smoke DPO/SFT training with inspectable artifacts (not claiming Qwen-scale LoRA/TRL quality).
- Evaluated TeacherBudget selection under a fixed teacher-call cap, reducing teacher calls by 80.0% vs label-all while measuring student task success and coverage (combined strategy, smoke oracle teachers).
- Shipped FastAPI artifact playback + portfolio export with measured metrics only (no fabricated results); full suite of unit/integration tests for Phases 1–7 smoke paths.

## Claims to avoid

- Frontier-scale / SOTA / production deployment
- Qwen-scale LoRA/TRL/GRPO quality from CPU smoke

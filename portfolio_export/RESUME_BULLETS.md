# Resume bullets (honest — unique Cohere alignment, not a clone)

Use **I built** if you created this independently.

## Pitch angle

Don’t mirror Cohere’s product surface. Show you independently identified a hard enterprise-agent problem they care about: **policy/SOP version shift** for tool-using agents — with eval, retrieval freshness, preference data, and a trainable shift split.

## Strong claims today

- I built PolicyShift, an evaluation environment for tool-using agents under evolving enterprise policies (3 domains × 3 sequential versions, 120+ executable cases, deterministic verifiers).
- I measured version-aware RAG vs baseline under matched smoke eval: task success 0.58 → 0.75; retrieval stale@5 0.45 → 0.00.
- I designed preference pairs (current-vs-stale, grounded-vs-unsupported, safe-vs-unsafe) and a **v1.0+v1.1 → v2.0** train/eval shift split for SFT/DPO experiments.
- I shipped artifacts, tests, and a public interactive demo/embed of the policy-shift failure mode.

## Do not claim yet (until GPU LoRA results are exported)

- Personal training/eval of Qwen LoRA with learning curves as the headline
- Distillation quality beyond smoke/replay
- That PolicyShift is “like Cohere North / Command”

## After a real `run_shift_experiment.py --no-smoke` (+ LoRA tool-loop eval)

Use only numbers from `artifacts/experiments/shift-study/summary.json`:

- I fine-tuned and preference-optimized an open-weight language model on versioned enterprise policies, evaluating generalization across a held-out policy shift against baseline and version-aware RAG agents.
- I compared Base / RAG / SFT / SFT+RAG / DPO / DPO+RAG on the same v2.0 cases with deterministic verifiers and PSRS; [paste PSRS + unsafe + stale rates].
- I published leakage-tested datasets, configs, seeds, hardware, and learning curves — no fabricated metrics.

Full copy kit: `docs/STAND_OUT_FOR_COHERE.md`

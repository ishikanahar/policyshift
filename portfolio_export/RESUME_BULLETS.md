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

## After a real `run_shift_experiment.py --no-smoke`

- I trained Qwen2.5-0.5B-Instruct (LoRA) SFT on pre-shift policies, evaluated under a v2.0 policy shift against RAG baselines, and ran DPO on verifier-derived preference pairs with seeded configs and published artifacts.

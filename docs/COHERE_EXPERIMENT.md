# Standing out to Cohere (without cloning Cohere)

**Not the goal:** recreate Command / North / Embed / Rerank.

**The goal:** show unique, credible alignment with what Cohere’s ML org actually cares about — enterprise agents, retrieval freshness, evaluation rigor, preference data, and trainable systems — through a problem Cohere does not already own as a demo: **tool-using agents under evolving enterprise SOPs (policy version shift).**

| Cohere theme | Your unique wedge in PolicyShift |
| --- | --- |
| Enterprise AI in regulated settings | Synthetic ops / lab / AI-governance SOPs that change over time |
| RAG + freshness / citations | Version-aware retrieval; stale@5 measurable under shift |
| Agents + tool use | Executable cases, tools, deterministic verifiers |
| Evaluation culture | Failure taxonomies, artifacts, matched baselines |
| Preference optimization | Current-vs-stale / safe-vs-unsafe pairs ready for DPO |
| Trainable models | Optional Qwen LoRA SFT→DPO on a **v1.x → v2.0** shift split |

Say **I built**. Lead with the evaluation + shift problem. Only call it post-training research after a real GPU run is published.

---

## Wired protocol (implemented)

```bash
# CPU smoke — proves the split + train filters + v2.0 eval wiring
python scripts/run_shift_experiment.py --smoke

# Real LoRA (GPU + pip install -e '.[training]')
python scripts/run_shift_experiment.py --no-smoke
```

What the script does:

1. `prepare_full_training_data.py --train-versions 1.0,1.1 --eval-versions 2.0`
2. SFT on pre-shift JSONL (`policy_version` stamped + filterable)
3. Evaluate baseline + version-aware RAG on held-out **v2.0** cases
4. DPO on preference pairs from train versions
5. Writes `artifacts/experiments/shift-v1-to-v2/summary.json`

Manual knobs:

```bash
python scripts/prepare_full_training_data.py \
  --train-versions 1.0,1.1 --eval-versions 2.0 --out-root data/shift

python scripts/train_sft.py --config configs/sft/full_gpu.yaml \
  --train-file data/shift/sft/sft_train.jsonl \
  --policy-versions 1.0,1.1 --no-smoke

python scripts/train_dpo.py --config configs/dpo/full_gpu.yaml \
  --train-file data/shift/dpo/dpo_train.jsonl \
  --policy-versions 1.0,1.1 --no-smoke
```

## Honest limits (keep these in interviews)

- Smoke SFT/DPO adapters are tiny / may replay teachers — not Qwen quality.
- Full `--no-smoke` trains real LoRA weights; **tool-loop eval of the LoRA student** is still the next hardening step (HF adapter exists; wire into harness when you have GPU results).
- Until then: claim **shift evaluation + preference data + trainable pipeline**, not “I shipped production post-training.”

## One-liner for applications

> I built PolicyShift to stress-test enterprise tool agents when company policies version — version-aware RAG, verifiers, and a v1→v2 train/eval split ready for SFT/DPO — because that’s the failure mode regulated copilots hit after SOP changes.

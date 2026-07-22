# What you add to stand out for a Cohere internship

Not more website polish. **One complete, reproducible post-training study** on a problem Cohere cares about but does not already own as a demo: tool agents under **versioned enterprise policy shift**.

---

## Exact headline result (hypothesis — replace with your real numbers)

> Preference optimization reduced unsafe tool actions, but it did not reliably update factual policy knowledge; combining DPO with version-aware retrieval produced the best safety–freshness tradeoff under a held-out policy change (train on v1.0+v1.1, eval on v2.0).

If your numbers contradict that, publish the actual outcome. A clean negative is stronger than a suspiciously perfect win.

---

## Exact sentence for applications / LinkedIn / email

**Before GPU run (honest today):**

> I built PolicyShift, a synthetic evaluation environment for enterprise tool agents under evolving SOPs — version-aware RAG, deterministic verifiers, preference-pair construction, and a leakage-tested v1→v2 train/eval split ready for SFT/DPO.

**After real `--no-smoke` (only then):**

> I fine-tuned and preference-optimized an open-weight language model on versioned enterprise policies, evaluating generalization across a held-out policy shift against baseline and version-aware RAG agents.

---

## Exact resume bullets to add after the study

Copy only verified metrics from `artifacts/experiments/shift-study/summary.json`.

1. I designed and ran a temporal policy-shift post-training study (train policies v1.0+v1.1; held-out eval v2.0) comparing Base, version-aware RAG, SFT, SFT+RAG, DPO, and DPO+RAG on the same cases with deterministic verifiers.
2. I trained Qwen2.5-[0.5B/1.5B]-Instruct with LoRA/QLoRA (SFT → DPO), published configs, seeds, dataset hashes, GPU/hardware, and learning curves — no fabricated metrics.
3. I showed [PSRS / unsafe-action / stale-citation] results: [fill from summary]; ablations included removing version metadata from retrieval and varying preference-data size.
4. I built hard-negative preference data (stale citations, unsafe tools, over-refusal, version ambiguity) with leakage tests forbidding v2.0 in train.

---

## Central research question (use this everywhere)

> When enterprise policies change, which adaptation strategy best preserves task performance while preventing stale-policy and unsafe tool actions?

---

## Why this stands out to Cohere (unique alignment, not a clone)

| They care about | You show independently |
| --- | --- |
| Enterprise agents / North-like workflows | Tool-using agents on ops, lab, AI-governance SOPs |
| RAG freshness & citations | Version-aware retrieval vs semantic-only under shift |
| Evaluation rigor | Deterministic verifiers + Policy Shift Robustness Score (PSRS) |
| Preference optimization | Hard negatives + DPO from SFT, not toy pairs |
| Trainable systems | Real LoRA SFT/DPO + manifests, not UI mock metrics |

**Do not say:** “I built something like Cohere Command/North.”  
**Do say:** “I stress-tested the failure mode regulated copilots hit when SOPs version.”

---

## Minimum bar that closes the credibility gap

Ship **all** of these with real artifacts (not smoke):

1. Real SFT + real DPO checkpoints  
2. Same held-out v2.0 eval for Base / RAG / SFT / SFT+RAG / DPO / DPO+RAG  
3. Automated **no v2.0 leakage** test (must pass)  
4. Metrics from saved traces (PSRS + bootstrap CIs)  
5. ≥3 ablations (e.g. no version metadata; no hard negatives; 25/50/100% prefs)  
6. Learning curves + hardware/config/commit published  
7. Site labels: **Real training run** vs **Smoke** vs **Illustrative**  
8. Failure case studies from real traces  

Run:

```bash
make setup
make data
# on GPU:
python scripts/run_shift_experiment.py --config configs/study/full.yaml --no-smoke
make report
```

Until that lands, PolicyShift is **strong infrastructure**. After it lands, it is a **credible post-training research project** for an ML internship interview.

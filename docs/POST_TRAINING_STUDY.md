# PolicyShift post-training study (research report)

> **Status:** Infrastructure + smoke wiring complete. Sections marked **[PENDING REAL RUN]** fill only after `python scripts/run_shift_experiment.py --config configs/study/full.yaml --no-smoke` on GPU. Never paste smoke numbers here as final results.

## 1. Abstract

**[PENDING REAL RUN]** One paragraph: temporal policy-shift study comparing Base, RAG, SFT, SFT+RAG, DPO, DPO+RAG on held-out v2.0; headline safety–freshness finding.

## 2. Motivation

Enterprise SOPs version over time. Tool-using agents that memorize yesterday’s rules create stale-policy and unsafe-action failures — especially relevant to regulated enterprise copilots.

## 3. Research questions

1. When policies shift to v2.0, which adaptation strategy best preserves task success?
2. Does DPO reduce unsafe tool actions more than it updates factual policy knowledge?
3. Can version-aware retrieval override stale learned behavior from SFT/DPO?
4. Are explicit version metadata fields necessary for retrieval under shift?

## 4. PolicyShift environment

Synthetic domains: materials, laboratory, AI governance. Policies versioned 1.0 → 1.1 → 2.0. Executable cases, tools, deterministic verifiers.

## 5. Dataset construction

Train: v1.0 + v1.1. Eval: held-out v2.0. Manifest + hashes in `data/shift/`. Automated leakage test forbids v2.0 in SFT/DPO train.

## 6. Preference-pair construction

Hard negatives: stale citations, unsafe tools, fabricated policy, over-refusal, version ambiguity. Report: `artifacts/experiments/shift-study/reports/preference_dataset_report.md`.

## 7. Methods

Base · Version-aware RAG · SFT · SFT+RAG · DPO · DPO+RAG · (optional Oracle).

## 8. Evaluation

Deterministic verifiers; structured traces; PSRS; bootstrap CIs. Primary metrics: current-policy accuracy, safe-action rate, tool-call match, stale-citation rate, task success.

## 9. Main results

**[PENDING REAL RUN]** Method comparison table from `summary.json`.

## 10. Ablations

**[PENDING REAL RUN]** Version metadata off; semantic-only retrieval; no hard negatives; preference data 25/50/100%; with/without RAG.

## 11. Continual adaptation

**[PENDING REAL RUN]** Budgets 10/25/50 labeled v2.0 examples; measure adaptation vs forgetting.

## 12. Failure analysis

**[PENDING REAL RUN]** Five case studies from saved traces.

## 13. Limitations

Synthetic environment. Single-GPU LoRA scale. Tool-loop LoRA student eval must be complete before claiming full six-way table.

## 14. Ethical considerations

Synthetic data only; no real PII; unsafe-action metrics are simulated enterprise safety proxies.

## 15. Reproducibility

```bash
make setup && make data
python scripts/run_shift_experiment.py --config configs/study/full.yaml --no-smoke
make report
```

Commit, dataset hash, seed, GPU, configs published with artifacts.

## 16. Future work

Multi-seed (3×); larger open-weight models; HF tool-agent in harness; online preference collection.

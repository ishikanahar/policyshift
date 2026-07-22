# Real training evidence (Colab T4)

**Date:** 2026-07-22  
**Model:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Hardware:** Google Colab NVIDIA T4  
**Mode:** `smoke: false` (real LoRA / TRL)

## Completed

| Run | Backend | Steps | Final loss | Checkpoint |
| --- | --- | --- | --- | --- |
| SFT | transformers + PEFT LoRA | 30 | 1.891 | `artifacts/experiments/sft-qwen05b/checkpoints/adapter/` |
| DPO | TRL DPO + PEFT LoRA | 30 | 0.6931 | `artifacts/experiments/dpo-qwen05b/checkpoints/adapter/` |

SFT loss fell from ~2.57 → ~1.68 during training (logged curve).  
DPO completed with saved `adapter_model.safetensors` + `train_metrics.json`.

## Honest limits

- Short runs (30 steps) — enough to prove **you personally trained** open-weight LoRA SFT/DPO, not enough for strong generalization claims.
- DPO reward margins were near zero in this short run — publish that honestly.
- Train data was not yet filtered to v1.0+v1.1 only (`policy_versions: null`).
- Full Base / RAG / SFT / DPO (± RAG) **tool-loop eval on held-out v2.0** is still open.

## Resume language you can use now

> I fine-tuned Qwen2.5-0.5B-Instruct with LoRA (SFT) and ran TRL DPO on preference pairs derived from PolicyShift’s enterprise policy cases, producing saved adapters and training metrics on a Colab T4 GPU.

## Resume language to wait on

> …evaluating generalization across a held-out policy shift against baseline and version-aware RAG…

Use that only after the v2.0 comparative eval is exported from traces.

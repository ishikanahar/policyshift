# Optional full GPU training

CPU smoke paths are the default and already produce measured portfolio metrics.
Use this guide when you have a GPU (Colab Pro / local NVIDIA) and want real Qwen LoRA / TRL runs.

## Install

```bash
pip install -e ".[training]"
# optional QLoRA:
pip install bitsandbytes
```

## 1) Prepare datasets

```bash
python scripts/prepare_full_training_data.py --n-cases 80 --out-root data/full
```

Writes:

- `data/full/sft/sft_train.jsonl`
- `data/full/dpo/dpo_train.jsonl`
- `data/full/rl/rl_pairs.jsonl`

## 2) SFT (LoRA on Qwen2.5-0.5B-Instruct)

```bash
python scripts/train_sft.py --config configs/sft/full_gpu.yaml \
  --train-file data/full/sft/sft_train.jsonl \
  --output-dir artifacts/experiments/sft-qwen05b/checkpoints \
  --no-smoke
```

## 3) DPO (TRL)

```bash
python scripts/train_dpo.py --config configs/dpo/full_gpu.yaml \
  --train-file data/full/dpo/dpo_train.jsonl \
  --output-dir artifacts/experiments/dpo-qwen05b/checkpoints \
  --no-smoke
```

## 4) RL smoke → full config

CPU RLOO-style smoke:

```bash
python scripts/run_phase7_smoke.py
```

Full GPU RL (GRPO-style stub via TRL when available):

```bash
python scripts/train_rl.py --config configs/rl/full_gpu.yaml --no-smoke
```

## Colab

Open `notebooks/PolicyShift_Full_GPU.ipynb`, set runtime to GPU, run all.

## Hardware notes

| Method | Rough VRAM |
| --- | --- |
| SFT LoRA 0.5B | ~8–12 GB |
| DPO LoRA 0.5B | ~12–16 GB |
| QLoRA 4-bit | often fits in 8 GB |

Do **not** paste Colab scores into README unless artifacts are exported into `artifacts/experiments/` and `scripts/export_portfolio.py` is re-run.

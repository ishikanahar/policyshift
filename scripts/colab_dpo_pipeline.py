#!/usr/bin/env python3
"""One-command Colab pipeline: data → (SFT if needed) → validate → DPO from SFT.

Usage on Colab after clone:
  %cd /content/policyshift
  !pip install -q -e '.[training]' && pip uninstall -y torchao
  !PYTHONPATH=src python scripts/colab_dpo_pipeline.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT_ADAPTER = ROOT / "artifacts/experiments/sft-qwen05b/checkpoints/adapter"
DPO_OUT = ROOT / "artifacts/experiments/dpo-qwen05b/checkpoints"


def run_py(args: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    cmd = [sys.executable, *args]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> None:
    run_py(
        [
            "scripts/prepare_full_training_data.py",
            "--n-cases",
            "80",
            "--out-root",
            "data/full",
        ]
    )

    if not (SFT_ADAPTER / "adapter_config.json").exists():
        print("No SFT adapter found — training SFT first (this is the slow step).")
        run_py(
            [
                "scripts/train_sft.py",
                "--config",
                "configs/sft/full_gpu.yaml",
                "--train-file",
                "data/full/sft/sft_train.jsonl",
                "--output-dir",
                str(SFT_ADAPTER.parent),
                "--no-smoke",
            ]
        )
    else:
        print(f"Using existing SFT adapter: {SFT_ADAPTER}")

    run_py(["scripts/validate_dpo_data.py", "--train-file", "data/full/dpo/dpo_train.jsonl"])

    run_py(
        [
            "scripts/train_dpo.py",
            "--config",
            "configs/dpo/full_gpu.yaml",
            "--train-file",
            "data/full/dpo/dpo_train.jsonl",
            "--output-dir",
            str(DPO_OUT),
            "--no-smoke",
        ]
    )

    print("\nDONE.")
    print(f"DPO adapter: {DPO_OUT / 'adapter'}")
    print(f"Metrics:     {DPO_OUT / 'train_metrics.json'}")
    print("Zip download path on Colab: /content/dpo_from_sft.zip")


if __name__ == "__main__":
    main()

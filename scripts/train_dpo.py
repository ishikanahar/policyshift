#!/usr/bin/env python3
"""Train DPO (smoke by default; full GPU via configs/dpo/full_gpu.yaml)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from policyshift.training.dpo_trainer import DPOTrainConfig, run_dpo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke/phase4_dpo.yaml"))
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    raw: dict = {}
    if args.config.exists():
        raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    train_file = args.train_file or Path(raw.get("train_file", "data/preferences/smoke/dpo_train.jsonl"))
    output_dir = args.output_dir or Path(raw.get("output_dir", "artifacts/experiments/dpo-smoke/checkpoints"))
    smoke = args.smoke if args.smoke is not None else bool(raw.get("smoke", True))

    if not train_file.exists():
        raise SystemExit(
            f"Missing {train_file}. Run scripts/generate_preference_pairs.py first, "
            "or scripts/train_dpo_smoke.py for the end-to-end Phase 4 smoke."
        )

    metrics = run_dpo(
        DPOTrainConfig(
            output_dir=str(output_dir),
            train_file=str(train_file),
            smoke=smoke,
            max_steps=int(raw.get("max_steps", 2)),
            learning_rate=float(raw.get("learning_rate", 1e-3)),
            beta=float(raw.get("beta", 0.1)),
            seed=int(raw.get("seed", 42)),
            model_name_or_path=str(raw.get("model_name_or_path", "smoke-tiny-dpo")),
            notes=str(raw.get("notes", "")),
        )
    )
    print(metrics)


if __name__ == "__main__":
    main()

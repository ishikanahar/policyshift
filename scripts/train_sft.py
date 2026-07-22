#!/usr/bin/env python3
"""Train SFT (smoke CPU or full GPU LoRA)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyshift.training.sft_trainer import TrainConfig, run_sft
from policyshift.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke/phase3_sft.yaml"))
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--policy-versions",
        type=str,
        default=None,
        help="Comma-separated versions to keep from train JSONL (e.g. 1.0,1.1).",
    )
    args = parser.parse_args()

    cfg_raw = load_yaml(args.config) if args.config.exists() else {}
    train_file = args.train_file or Path(
        str(cfg_raw.get("train_file", "data/generated/sft/sft_train.jsonl"))
    )
    output_dir = args.output_dir or Path(
        str(cfg_raw.get("output_dir", "artifacts/experiments/sft-smoke/checkpoints"))
    )
    smoke = bool(cfg_raw.get("smoke", True)) if args.smoke is None else bool(args.smoke)
    max_steps = args.max_steps if args.max_steps is not None else int(cfg_raw.get("max_steps", 2))

    from policyshift.training.version_filters import parse_policy_versions

    policy_versions = parse_policy_versions(
        args.policy_versions or cfg_raw.get("policy_versions")
    )

    if not train_file.exists():
        raise SystemExit(
            f"Missing train file {train_file}. Run scripts/generate_teacher_trajectories.py "
            "and build SFT data first, or run scripts/train_sft.py via phase3 smoke."
        )

    metrics = run_sft(
        TrainConfig(
            output_dir=str(output_dir),
            train_file=str(train_file),
            smoke=bool(smoke),
            max_steps=max_steps,
            model_name_or_path=str(cfg_raw.get("model_name_or_path", "smoke-tiny-policylm")),
            learning_rate=float(cfg_raw.get("learning_rate", 1e-3)),
            seed=int(cfg_raw.get("seed", 42)),
            lora_r=int(cfg_raw.get("lora_r", 4)),
            lora_alpha=int(cfg_raw.get("lora_alpha", 8)),
            notes=str(cfg_raw.get("notes", "")),
            policy_versions=policy_versions,
        )
    )
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()

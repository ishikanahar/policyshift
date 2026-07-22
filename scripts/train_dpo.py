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
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--policy-versions",
        type=str,
        default=None,
        help="Comma-separated versions to keep from DPO JSONL (e.g. 1.0,1.1).",
    )
    args = parser.parse_args()

    raw: dict = {}
    if args.config.exists():
        raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    train_file = args.train_file or Path(raw.get("train_file", "data/preferences/smoke/dpo_train.jsonl"))
    output_dir = args.output_dir or Path(raw.get("output_dir", "artifacts/experiments/dpo-smoke/checkpoints"))
    smoke = args.smoke if args.smoke is not None else bool(raw.get("smoke", True))
    if args.max_steps is not None:
        max_steps: int | None = args.max_steps
    elif "max_steps" in raw:
        raw_ms = raw.get("max_steps")
        max_steps = None if raw_ms in (None, "null", -1) else int(raw_ms)
    else:
        max_steps = 2 if smoke else None

    from policyshift.training.version_filters import parse_policy_versions

    policy_versions = parse_policy_versions(args.policy_versions or raw.get("policy_versions"))

    if not train_file.exists():
        raise SystemExit(
            f"Missing {train_file}. Run scripts/generate_preference_pairs.py first, "
            "or scripts/train_dpo_smoke.py for the end-to-end Phase 4 smoke."
        )

    sft_adapter = raw.get("sft_adapter_path") or raw.get("init_from")
    if not smoke and not sft_adapter:
        raise SystemExit(
            "Full DPO requires sft_adapter_path / init_from in the config "
            "(initialize from SFT checkpoint, not raw Qwen)."
        )

    metrics = run_dpo(
        DPOTrainConfig(
            output_dir=str(output_dir),
            train_file=str(train_file),
            smoke=smoke,
            max_steps=max_steps,
            num_train_epochs=float(raw.get("num_train_epochs", 1.0)),
            learning_rate=float(raw.get("learning_rate", 1e-3)),
            beta=float(raw.get("beta", 0.1)),
            seed=int(raw.get("seed", 42)),
            model_name_or_path=str(
                raw.get("model_name_or_path") or raw.get("base_model") or "smoke-tiny-dpo"
            ),
            sft_adapter_path=str(sft_adapter) if sft_adapter else None,
            notes=str(raw.get("notes", "")),
            policy_versions=policy_versions,
            max_seq_length=int(raw.get("max_seq_length", raw.get("max_length", 1536))),
            max_prompt_length=int(raw.get("max_prompt_length", 1152)),
            max_completion_length=int(raw.get("max_completion_length", 384)),
            lora_r=int(raw.get("lora_r", 4)),
            lora_alpha=int(raw.get("lora_alpha", 8)),
            per_device_train_batch_size=int(raw.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
            gradient_checkpointing=bool(raw.get("gradient_checkpointing", False)),
        )
    )
    print(metrics)


if __name__ == "__main__":
    main()

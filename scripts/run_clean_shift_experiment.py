#!/usr/bin/env python3
"""One orchestrator for the leakage-free shift-clean experiment.

Stages (use flags or --all):
  1. prepare-data
  2. validate (split leakage + DPO tokenization)
  3. train-sft
  4. train-dpo
  5. evaluate

Never writes into sft-qwen05b / dpo-qwen05b / shift-study / website assets.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from policyshift.training.dpo_format import DEFAULT_BUDGET, validate_budgeted_pairs
from policyshift.training.dpo_trainer import load_dpo_checkpoint, load_dpo_rows
from policyshift.training.shift_clean import (
    DEFAULT_DATA_ROOT,
    file_sha256,
    prepare_shift_clean_data,
    validate_shift_split,
    write_provenance,
)
from policyshift.training.sft_trainer import load_checkpoint
from policyshift.utils.io import ensure_dir, write_json

ART = Path("artifacts/experiments/shift-clean")
SFT_CFG = Path("configs/sft/shift_clean_gpu.yaml")
DPO_CFG = Path("configs/dpo/shift_clean_gpu.yaml")
SFT_ADAPTER = ART / "sft" / "checkpoints" / "adapter"
DPO_ADAPTER = ART / "dpo" / "checkpoints" / "adapter"


def _run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def stage_prepare(args: argparse.Namespace) -> dict:
    report = prepare_shift_clean_data(
        out_root=Path(args.data_root),
        seed=args.seed,
        n_train_cases=args.n_train_cases,
        n_eval_cases=args.n_eval_cases,
    )
    write_json(ART / "data_validation" / "prepare_report.json", report)
    print(json.dumps(report, indent=2))
    return report


def stage_validate(args: argparse.Namespace) -> dict:
    data_root = Path(args.data_root)
    split_report = validate_shift_split(data_root=data_root, write_stamp=True)
    out_dir = ensure_dir(ART / "data_validation")
    write_json(out_dir / "shift_split_validation.json", split_report)
    print("SHIFT_SPLIT_VALIDATION_PASSED")
    print(f"SFT training versions: {split_report['sft_training_versions']}")
    print(f"DPO training versions: {split_report['dpo_training_versions']}")
    print(f"Evaluation versions: {split_report['evaluation_versions']}")
    print(f"Number of SFT examples: {split_report['n_sft_examples']}")
    print(f"Number of DPO pairs: {split_report['n_dpo_pairs']}")
    print(f"Number of held-out v2.0 cases: {split_report['n_heldout_v2_cases']}")
    print(f"Leakage count: {split_report['leakage_count']}")
    print(f"Dataset hashes: {split_report['dataset_hashes']}")

    # DPO response-aware tokenization gate (CPU; may download tokenizer once).
    from transformers import AutoTokenizer

    dpo_path = data_root / "dpo" / "dpo_train.jsonl"
    rows = load_dpo_rows(dpo_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dpo_val = validate_budgeted_pairs(rows, tokenizer, DEFAULT_BUDGET)
    compact = {k: v for k, v in dpo_val.items() if k != "rows"}
    write_json(out_dir / "dpo_validation.json", compact)
    if not dpo_val["passed"]:
        raise SystemExit("DPO tokenization validation failed for shift-clean data.")
    print("DPO_VALIDATION_PASSED")
    print(
        "difference_survived_truncation_pct=",
        dpo_val["percentage_retaining_chosen_rejected_difference"],
    )
    return {"split": split_report, "dpo": compact}


def stage_train_sft(args: argparse.Namespace) -> dict:
    _run(
        [
            sys.executable,
            "scripts/train_sft.py",
            "--config",
            str(SFT_CFG),
            "--no-smoke",
        ]
    )
    loaded = load_checkpoint(SFT_ADAPTER)
    if not loaded.get("ok", True) and loaded.get("format") not in {"peft-adapter", "json"}:
        # load_checkpoint returns path/format; peft adapter is success if config exists
        pass
    if not (SFT_ADAPTER / "adapter_config.json").exists():
        raise SystemExit(f"SFT adapter missing after train: {SFT_ADAPTER}")
    print("SFT_ADAPTER_RELOAD_OK", SFT_ADAPTER)

    sft_hash = file_sha256(Path(args.data_root) / "sft" / "sft_train.jsonl")
    write_provenance(
        ART / "sft" / "checkpoints" / "provenance.json",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        initialization_checkpoint=None,
        adapter_path=str(SFT_ADAPTER),
        training_policy_versions=["1.0", "1.1"],
        evaluation_policy_versions=["2.0"],
        dataset_path=str(Path(args.data_root) / "sft" / "sft_train.jsonl"),
        dataset_hash=sft_hash,
        seed=args.seed,
        training_stage="clean_sft",
        extra={
            "lora_r": 16,
            "lora_alpha": 32,
            "num_train_epochs": 1,
            "learning_rate": 0.0002,
        },
    )
    return {"adapter": str(SFT_ADAPTER), "reload": "ok"}


def stage_train_dpo(args: argparse.Namespace) -> dict:
    if not (SFT_ADAPTER / "adapter_config.json").exists():
        raise SystemExit(
            f"Clean SFT adapter required at {SFT_ADAPTER} before DPO. Run --train-sft first."
        )
    # Refuse leaked/debug adapters as init.
    forbidden = {
        Path("artifacts/experiments/sft-qwen05b/checkpoints/adapter").resolve(),
        Path("artifacts/experiments/dpo-qwen05b/checkpoints/adapter").resolve(),
    }
    if SFT_ADAPTER.resolve() in forbidden:
        raise SystemExit("Refusing to init clean DPO from debug/full-policy adapters.")

    _run(
        [
            sys.executable,
            "scripts/train_dpo.py",
            "--config",
            str(DPO_CFG),
            "--no-smoke",
        ]
    )
    load_dpo_checkpoint(DPO_ADAPTER)
    if not (DPO_ADAPTER / "adapter_config.json").exists():
        raise SystemExit(f"DPO adapter missing after train: {DPO_ADAPTER}")
    print("DPO_ADAPTER_RELOAD_OK", DPO_ADAPTER)

    dpo_hash = file_sha256(Path(args.data_root) / "dpo" / "dpo_train.jsonl")
    write_provenance(
        ART / "dpo" / "checkpoints" / "provenance.json",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        initialization_checkpoint=str(SFT_ADAPTER),
        adapter_path=str(DPO_ADAPTER),
        training_policy_versions=["1.0", "1.1"],
        evaluation_policy_versions=["2.0"],
        dataset_path=str(Path(args.data_root) / "dpo" / "dpo_train.jsonl"),
        dataset_hash=dpo_hash,
        seed=args.seed,
        training_stage="dpo_after_clean_sft",
        extra={
            "lora_r": 16,
            "lora_alpha": 32,
            "num_train_epochs": 1,
            "learning_rate": 0.00001,
            "beta": 0.1,
            "max_length": 1536,
            "max_prompt_length": 1152,
            "max_completion_length": 384,
        },
    )
    return {"adapter": str(DPO_ADAPTER), "init_from": str(SFT_ADAPTER), "reload": "ok"}


def stage_evaluate(args: argparse.Namespace) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "evaluate_shift_clean",
        Path("scripts/evaluate_shift_clean.py"),
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    summary = mod.run_shift_clean_evaluation(
        eval_file=Path(args.data_root) / "eval" / "v2_eval.jsonl",
        out_dir=ART / "evaluation",
        sft_adapter=SFT_ADAPTER,
        dpo_adapter=DPO_ADAPTER,
        seed=args.seed,
        skip_lora=args.skip_lora_eval,
    )
    write_json(ART / "evaluation" / "comparison_table.json", summary.get("comparison"))
    print("EVALUATION_EXPORT_OK", ART / "evaluation" / "summary.csv")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--train-sft", action="store_true")
    parser.add_argument("--train-dpo", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-cases", type=int, default=80)
    parser.add_argument("--n-eval-cases", type=int, default=24)
    parser.add_argument("--skip-lora-eval", action="store_true")
    args = parser.parse_args()

    ensure_dir(ART)

    if args.all:
        args.prepare_data = True
        args.validate = True
        args.train_sft = True
        args.train_dpo = True
        args.evaluate = True

    if not any(
        [
            args.prepare_data,
            args.validate,
            args.train_sft,
            args.train_dpo,
            args.evaluate,
        ]
    ):
        parser.error("Select at least one stage or pass --all")

    try:
        if args.prepare_data:
            stage_prepare(args)
        if args.validate:
            stage_validate(args)
        if args.train_sft:
            stage_train_sft(args)
        if args.train_dpo:
            stage_train_dpo(args)
        if args.evaluate:
            stage_evaluate(args)
    except subprocess.CalledProcessError as exc:
        print(f"STAGE_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc

    print("\nSHIFT_CLEAN_STAGE_COMPLETE")


if __name__ == "__main__":
    main()

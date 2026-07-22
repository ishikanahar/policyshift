#!/usr/bin/env python3
"""Validate DPO pairs under production response-aware token budgets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from policyshift.training.dpo_format import (
    DEFAULT_BUDGET,
    DPOBudgetConfig,
    validate_budgeted_pairs,
)
from policyshift.training.dpo_trainer import load_dpo_rows
from policyshift.utils.io import ensure_dir, write_json

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TRAIN = Path("data/full/dpo/dpo_train.jsonl")
DEFAULT_OUT = Path("artifacts/data_validation/dpo_validation.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=DEFAULT_BUDGET.max_length)
    parser.add_argument("--max-prompt-length", type=int, default=DEFAULT_BUDGET.max_prompt_length)
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=DEFAULT_BUDGET.max_completion_length,
    )
    args = parser.parse_args()

    if not args.train_file.exists():
        raise SystemExit(
            f"Missing {args.train_file}. Run:\n"
            "  python scripts/prepare_full_training_data.py --n-cases 80 --out-root data/full"
        )

    budget = DPOBudgetConfig(
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
    )

    rows = load_dpo_rows(args.train_file)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    report = validate_budgeted_pairs(
        rows,
        tokenizer,
        budget,
        n_samples=args.n_samples,
    )
    report["train_file"] = str(args.train_file)
    report["model_name_or_path"] = args.model
    # Compact on-disk report: keep samples + aggregates, drop full per-row dump by default size.
    disk = {k: v for k, v in report.items() if k != "rows"}
    disk["n_row_reports"] = len(report.get("rows", []))

    ensure_dir(args.out.parent)
    write_json(args.out, disk)

    printable = {
        k: v
        for k, v in disk.items()
        if k not in {"sample_pairs"}
    }
    print(json.dumps(printable, indent=2))
    print("\n=== Sample pairs (first differing completion span) ===")
    for i, sample in enumerate(report["sample_pairs"], 1):
        print(f"\n[{i}] id={sample.get('id')} case={sample.get('case_id')}")
        print(f"  first_differing_completion_token={sample.get('first_differing_completion_token')}")
        print(f"  difference_survived_truncation={sample.get('difference_survived_truncation')}")
        print(f"  prompt_tokens={sample.get('prompt_tokens')} "
              f"chosen_loss={sample.get('chosen_loss_tokens')} "
              f"rejected_loss={sample.get('rejected_loss_tokens')}")
        print(f"  chosen_span:   {sample.get('chosen_span')!r}")
        print(f"  rejected_span: {sample.get('rejected_span')!r}")

    print(f"\nWrote {args.out}")
    if not report["passed"]:
        print(
            "FAIL: DPO validation requirements not met "
            "(100% raw diffs, >=95% tokenized diffs after budgeting, "
            "non-empty loss masks, differing completion token in loss).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("DPO_VALIDATION_PASSED")


if __name__ == "__main__":
    main()

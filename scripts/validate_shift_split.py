#!/usr/bin/env python3
"""Hard leakage gate for the clean temporal policy-shift split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from policyshift.training.shift_clean import DEFAULT_DATA_ROOT, validate_shift_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/experiments/shift-clean/data_validation/shift_split_validation.json"),
    )
    args = parser.parse_args()

    try:
        report = validate_shift_split(data_root=args.data_root, write_stamp=True)
    except AssertionError as exc:
        print(f"SHIFT_SPLIT_VALIDATION_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("SHIFT_SPLIT_VALIDATION_PASSED")
    print(f"SFT training versions: {report['sft_training_versions']}")
    print(f"DPO training versions: {report['dpo_training_versions']}")
    print(f"Evaluation versions: {report['evaluation_versions']}")
    print(f"Number of SFT examples: {report['n_sft_examples']}")
    print(f"Number of DPO pairs: {report['n_dpo_pairs']}")
    print(f"Number of held-out v2.0 cases: {report['n_heldout_v2_cases']}")
    print(f"Leakage count: {report['leakage_count']}")
    print(f"Dataset hashes: {report['dataset_hashes']}")


if __name__ == "__main__":
    main()

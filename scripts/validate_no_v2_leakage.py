#!/usr/bin/env python3
"""Fail if SFT/DPO train JSONL contains held-out policy versions (default 2.0)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyshift.training.leakage import validate_shift_datasets
from policyshift.training.version_filters import parse_policy_versions
from policyshift.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--dpo", type=Path, required=True)
    parser.add_argument("--forbidden", type=str, default="2.0")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    forbidden = set(parse_policy_versions(args.forbidden) or [])
    report = validate_shift_datasets(
        sft_path=args.sft,
        dpo_path=args.dpo,
        forbidden_versions=forbidden,
    )
    if args.out:
        write_json(args.out, report)
    print(json.dumps(report, indent=2))
    print("LEAKAGE_CHECK_PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare leakage-free shift-clean datasets under data/shift_clean/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyshift.training.shift_clean import DEFAULT_DATA_ROOT, prepare_shift_clean_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-cases", type=int, default=80)
    parser.add_argument("--n-eval-cases", type=int, default=24)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    report = prepare_shift_clean_data(
        out_root=args.out_root,
        seed=args.seed,
        n_train_cases=args.n_train_cases,
        n_eval_cases=args.n_eval_cases,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate deterministic synthetic cases and leakage report."""

from __future__ import annotations

import argparse
from pathlib import Path

from policyshift.data_generation.cases import check_split_leakage, generate_cases, write_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cases", type=int, default=120)
    parser.add_argument("--out", type=Path, default=Path("data/generated/cases"))
    args = parser.parse_args()
    cases = generate_cases(seed=args.seed, n_cases=args.n_cases)
    write_cases(cases, args.out)
    report = check_split_leakage(cases)
    print(f"Wrote {len(cases)} cases; leakage_ok={report['ok']}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate synthetic versioned policy documents."""

from __future__ import annotations

import argparse
from pathlib import Path

from policyshift.data_generation.policies import write_policies
from policyshift.schemas import export_json_schemas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("policies"))
    parser.add_argument("--export-json", type=Path, default=Path("data/generated/policies"))
    args = parser.parse_args()
    paths = write_policies(args.out, export_json_dir=args.export_json)
    export_json_schemas(Path("policies/schemas"))
    print(f"Wrote {len(paths)} policies")


if __name__ == "__main__":
    main()

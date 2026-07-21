#!/usr/bin/env python3
"""Export portfolio_export + resume bullets + technical report from measured artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyshift.portfolio import write_portfolio_export


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("portfolio_export"))
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/experiments"))
    args = p.parse_args()
    paths = write_portfolio_export(args.out, artifact_root=args.artifact_root)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    resume = (args.out / "RESUME_BULLETS.md").read_text(encoding="utf-8")
    print(resume)


if __name__ == "__main__":
    main()

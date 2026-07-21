#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyshift.training.rl_smoke import run_phase7_smoke


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 7 RL smoke")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-cases", type=int, default=40)
    p.add_argument("--n-eval", type=int, default=12)
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/experiments"))
    p.add_argument("--experiment-id", type=str, default="phase7-smoke-local")
    args = p.parse_args()
    result = run_phase7_smoke(
        seed=args.seed,
        n_cases=args.n_cases,
        n_eval=args.n_eval,
        artifact_root=args.artifact_root,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(result["summary"]["conditions"], indent=2))
    print(json.dumps(result["summary"]["reward_hacking"], indent=2))
    print(json.dumps({"experiment_id": result["experiment_id"]}, indent=2))


if __name__ == "__main__":
    main()

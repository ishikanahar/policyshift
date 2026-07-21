#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyshift.training.teacher_budget import run_phase6_smoke


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 6 TeacherBudget smoke")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-cases", type=int, default=60)
    p.add_argument("--n-eval", type=int, default=12)
    p.add_argument("--budget", type=int, default=12)
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/experiments"))
    p.add_argument("--experiment-id", type=str, default="phase6-smoke-local")
    args = p.parse_args()
    result = run_phase6_smoke(
        seed=args.seed,
        n_cases=args.n_cases,
        n_eval=args.n_eval,
        budget=args.budget,
        artifact_root=args.artifact_root,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(result["summary"]["comparison"], indent=2))
    print(json.dumps({"experiment_id": result["experiment_id"], "paths": result["paths"]}, indent=2))


if __name__ == "__main__":
    main()

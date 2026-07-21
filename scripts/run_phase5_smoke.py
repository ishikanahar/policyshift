#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyshift.training.continual import run_phase5_smoke


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 5 continual learning smoke")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-cases", type=int, default=90)
    p.add_argument("--per-version-eval", type=int, default=6)
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/experiments"))
    p.add_argument("--experiment-id", type=str, default="phase5-smoke-local")
    args = p.parse_args()
    result = run_phase5_smoke(
        seed=args.seed,
        n_cases=args.n_cases,
        per_version_eval=args.per_version_eval,
        artifact_root=args.artifact_root,
        experiment_id=args.experiment_id,
    )
    print(json.dumps({"experiment_id": result["experiment_id"], "paths": result["paths"]}, indent=2))
    for name, s in result["summary"]["strategies"].items():
        print(
            f"{name}: forgetting={s['average_forgetting']:.3f} "
            f"bwt={s['average_backward_transfer']:.3f} stale={s['mean_stale_policy_error']:.3f}"
        )


if __name__ == "__main__":
    main()

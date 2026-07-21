#!/usr/bin/env python3
"""End-to-end Phase 4 smoke: preference pairs → DPO → base/RAG/DPO compare."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyshift.training import run_phase4_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cases", type=int, default=40)
    parser.add_argument("--n-eval", type=int, default=12)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--experiment-id", type=str, default=None)
    args = parser.parse_args()

    result = run_phase4_smoke(
        seed=args.seed,
        n_cases=args.n_cases,
        n_eval=args.n_eval,
        artifact_root=args.artifact_root,
        experiment_id=args.experiment_id,
    )
    print(json.dumps({"experiment_id": result["experiment_id"], "paths": result["paths"]}, indent=2))
    print(json.dumps(result["summary"]["conditions"], indent=2))
    print(
        json.dumps(
            {
                "n_pairs": result["summary"]["preferences"]["n_pairs"],
                "checkpoint": result["summary"]["checkpoint_loaded"]["path"],
                "train_backend": result["summary"]["training"]["training"]["backend"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

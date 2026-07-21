#!/usr/bin/env python3
"""Run Phase 2 evaluation: retrieval ablations + base/RAG agents + artifact export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyshift.evaluation import run_phase2_smoke
from policyshift.schemas import Split
from policyshift.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke/phase2.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-cases", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--experiment-id", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config) if args.config.exists() else {}
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    n_cases = args.n_cases if args.n_cases is not None else int(cfg.get("n_cases", 40))
    split_name = args.split or str(cfg.get("split", "validation"))
    artifact_root = args.artifact_root or Path(str(cfg.get("artifact_root", "artifacts/experiments")))

    result = run_phase2_smoke(
        seed=seed,
        n_cases=n_cases,
        split=Split(split_name),
        artifact_root=artifact_root,
        experiment_id=args.experiment_id,
    )
    print(json.dumps({"experiment_id": result["experiment_id"], "paths": result["paths"]}, indent=2))
    print(json.dumps(result["summary"]["conditions"], indent=2))
    print(json.dumps(result["summary"]["retrieval_ablation"], indent=2))


if __name__ == "__main__":
    main()

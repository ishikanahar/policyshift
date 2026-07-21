#!/usr/bin/env python3
"""Validate trajectories with deterministic verifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.schemas import AgentTrajectory
from policyshift.training.teacher import load_trajectories_jsonl
from policyshift.verification.verifiers import TrajectoryVerifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cases", type=int, default=120)
    args = parser.parse_args()

    store = PolicyStore.from_builtin()
    verifier = TrajectoryVerifier(store)
    cases = {c.case_id: c for c in generate_cases(seed=args.seed, n_cases=args.n_cases)}
    trajs = load_trajectories_jsonl(args.trajectories)
    ok = 0
    for traj in trajs:
        case = cases.get(traj.case_id)
        if case is None:
            print(json.dumps({"trajectory_id": traj.trajectory_id, "error": "unknown_case"}))
            continue
        results = verifier.verify(case, traj)
        success = verifier.success(results)
        ok += int(success)
        if not success:
            print(
                json.dumps(
                    {
                        "trajectory_id": traj.trajectory_id,
                        "case_id": traj.case_id,
                        "success": False,
                        "failed": [r.model_dump() for r in results if not r.passed],
                    }
                )
            )
    print(json.dumps({"n": len(trajs), "passed": ok}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate verifier-filtered teacher trajectories for distillation."""

from __future__ import annotations

import argparse
from pathlib import Path

from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.schemas import Split
from policyshift.training.teacher import TeacherTrajectoryGenerator, write_teacher_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cases", type=int, default=40)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--source", type=str, default="oracle", choices=["oracle", "file"])
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/generated/teacher"))
    args = parser.parse_args()

    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=args.seed, n_cases=max(args.n_cases, 120))
    split = Split(args.split)
    selected = [c for c in cases if c.split == split][: args.n_cases]
    gen = TeacherTrajectoryGenerator(store, source=args.source, file_path=args.file)
    accepted, report = gen.generate_batch(selected)
    paths = write_teacher_artifacts(args.out, accepted, report)
    print(
        f"accepted={report.n_accepted}/{report.n_cases} "
        f"tokens={report.total_tokens} out={paths['accepted']}"
    )


if __name__ == "__main__":
    main()

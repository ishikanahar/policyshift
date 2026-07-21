#!/usr/bin/env python3
"""Generate preference pairs and DPO JSONL from verified oracle trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.schemas import Split
from policyshift.training.preferences import (
    build_preference_dataset,
    pairs_to_dpo_examples,
    write_preference_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cases", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("data/preferences/smoke"))
    args = parser.parse_args()

    store = PolicyStore.from_builtin()
    cases = [c for c in generate_cases(seed=args.seed, n_cases=max(args.n_cases, 120)) if c.split == Split.TRAIN][
        : args.n_cases
    ]
    dataset = build_preference_dataset(cases, policy_store=store)
    traj_map = {t.trajectory_id: t for t in dataset["chosen"] + dataset["rejected"]}
    examples = pairs_to_dpo_examples(cases, dataset["pairs"], traj_map)
    paths = write_preference_artifacts(
        args.out,
        pairs=dataset["pairs"],
        chosen=dataset["chosen"],
        rejected=dataset["rejected"],
        dpo_examples=examples,
        skipped=dataset["skipped"],
    )
    print(f"Wrote {dataset['n_pairs']} pairs → {paths['explorer']}")
    print(f"DPO train: {paths['dpo_jsonl']} ({len(examples)} examples)")


if __name__ == "__main__":
    main()

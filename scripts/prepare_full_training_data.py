#!/usr/bin/env python3
"""Prepare SFT + DPO (+ RL pair) datasets for optional full GPU training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.schemas import Split
from policyshift.training.preferences import (
    build_preference_dataset,
    pairs_to_dpo_examples,
    write_preference_artifacts,
)
from policyshift.training.sft_data import build_sft_dataset, write_sft_dataset
from policyshift.training.teacher import TeacherTrajectoryGenerator, write_teacher_artifacts
from policyshift.utils.io import ensure_dir, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cases", type=int, default=80)
    parser.add_argument("--out-root", type=Path, default=Path("data/full"))
    args = parser.parse_args()

    store = PolicyStore.from_builtin()
    all_cases = generate_cases(seed=args.seed, n_cases=max(args.n_cases, 120))
    train_cases = [c for c in all_cases if c.split == Split.TRAIN][: args.n_cases]

    root = ensure_dir(args.out_root)
    teacher_dir = ensure_dir(root / "teacher")
    sft_dir = ensure_dir(root / "sft")
    dpo_dir = ensure_dir(root / "dpo")
    rl_dir = ensure_dir(root / "rl")

    # Teacher → SFT
    gen = TeacherTrajectoryGenerator(store, source="oracle")
    accepted, report = gen.generate_batch(train_cases)
    write_teacher_artifacts(teacher_dir, accepted, report)
    examples = build_sft_dataset(train_cases, accepted)
    sft_paths = write_sft_dataset(examples, sft_dir)

    # Preference / DPO
    prefs = build_preference_dataset(train_cases, policy_store=store)
    traj_map = {t.trajectory_id: t for t in prefs["chosen"] + prefs["rejected"]}
    dpo_examples = pairs_to_dpo_examples(train_cases, prefs["pairs"], traj_map)
    dpo_paths = write_preference_artifacts(
        dpo_dir,
        pairs=prefs["pairs"],
        chosen=prefs["chosen"],
        rejected=prefs["rejected"],
        dpo_examples=dpo_examples,
        skipped=prefs["skipped"],
    )

    # RL preference-like pairs (good oracle vs weaker baseline finals)
    from policyshift.agents.baseline import BaselineAgent

    baseline = BaselineAgent(store)
    rl_rows = []
    for case in train_cases:
        good = next((t for t in accepted if t.case_id == case.case_id), None)
        if good is None:
            continue
        bad = baseline.resolve(case)
        rl_rows.append(
            {
                "case_id": case.case_id,
                "prompt": case.event_type,
                "chosen": good.final_answer,
                "rejected": bad.final_answer,
                "reward_good": good.total_reward,
                "reward_bad": bad.total_reward,
            }
        )
    rl_path = write_jsonl(rl_dir / "rl_pairs.jsonl", rl_rows)

    manifest = {
        "n_cases": len(train_cases),
        "n_teacher_accepted": len(accepted),
        "n_sft": len(examples),
        "n_dpo": len(dpo_examples),
        "n_rl_pairs": len(rl_rows),
        "paths": {
            "sft": str(sft_paths["jsonl"]),
            "dpo": str(dpo_paths["dpo_jsonl"]),
            "rl": str(rl_path),
            "teacher_report": str(teacher_dir / "teacher_report.json"),
        },
    }
    write_json(root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

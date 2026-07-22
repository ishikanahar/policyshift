#!/usr/bin/env python3
"""Policy-shift experiment: train on v1.x, evaluate on v2.0.

Default mode is CPU smoke (reproducible wiring). Pass --no-smoke with GPU +
`pip install -e '.[training]'` for real Qwen LoRA SFT/DPO.

Unique angle (not a Cohere clone): enterprise tool agents fail when SOPs
version — measurable retrieval + preference + post-training under that shift.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.evaluation.harness import _select_cases, run_agent_evaluation, run_retrieval_ablation
from policyshift.evaluation.metrics import aggregate_metrics
from policyshift.schemas import Split
from policyshift.training.dpo_trainer import DPOTrainConfig, run_dpo
from policyshift.training.sft_trainer import TrainConfig, run_sft
from policyshift.training.version_filters import parse_policy_versions
from policyshift.utils.io import ensure_dir, write_json


def _summarize_agent(name: str, cases, store) -> dict:
    trajs, rows = run_agent_evaluation(name, cases, policy_store=store)
    return {
        "agent": name,
        "n": len(cases),
        "aggregate": aggregate_metrics(rows),
        "n_success": sum(1 for t in trajs if t.success),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=48)
    parser.add_argument("--n-eval", type=int, default=24)
    parser.add_argument("--train-versions", type=str, default="1.0,1.1")
    parser.add_argument("--eval-versions", type=str, default="2.0")
    parser.add_argument("--out-root", type=Path, default=Path("data/shift"))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/experiments/shift-v1-to-v2"),
    )
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    train_versions = parse_policy_versions(args.train_versions)
    eval_versions = parse_policy_versions(args.eval_versions)
    assert train_versions and eval_versions

    out_root = ensure_dir(args.out_root)
    art = ensure_dir(args.artifact_root)

    # 1) Prepare version-filtered datasets
    prep = [
        sys.executable,
        "scripts/prepare_full_training_data.py",
        "--seed",
        str(args.seed),
        "--n-cases",
        str(args.n_train),
        "--out-root",
        str(out_root),
        "--train-versions",
        ",".join(train_versions),
        "--eval-versions",
        ",".join(eval_versions),
    ]
    subprocess.check_call(prep)

    sft_file = out_root / "sft" / "sft_train.jsonl"
    dpo_file = out_root / "dpo" / "dpo_train.jsonl"

    # 2) SFT on pre-shift policies
    sft_metrics = run_sft(
        TrainConfig(
            output_dir=str(art / "sft" / "checkpoints"),
            train_file=str(sft_file),
            smoke=args.smoke,
            max_steps=2 if args.smoke else 200,
            policy_versions=train_versions,
            notes=f"shift-sft train_versions={train_versions}",
            model_name_or_path="smoke-tiny-policylm" if args.smoke else "Qwen/Qwen2.5-0.5B-Instruct",
        )
    )

    # 3) Eval baselines on held-out shifted policies (v2.0)
    store = PolicyStore.from_builtin()
    all_cases = generate_cases(seed=args.seed, n_cases=max(args.n_train, args.n_eval, 120))
    eval_cases = _select_cases(
        all_cases,
        split=Split.VALIDATION,
        limit=args.n_eval,
        policy_versions=eval_versions,
    )
    if len(eval_cases) < max(1, args.n_eval // 2):
        # Top up from all splits if validation is thin for that version
        eval_cases = _select_cases(
            all_cases,
            split=None,
            limit=args.n_eval,
            policy_versions=eval_versions,
        )

    retrieval = run_retrieval_ablation(eval_cases, policy_store=store)
    baseline = _summarize_agent("baseline", eval_cases, store)
    rag = _summarize_agent("rag", eval_cases, store)

    # 4) DPO on preference pairs from train versions
    dpo_metrics = run_dpo(
        DPOTrainConfig(
            output_dir=str(art / "dpo" / "checkpoints"),
            train_file=str(dpo_file),
            smoke=args.smoke,
            max_steps=2 if args.smoke else 200,
            policy_versions=train_versions,
            notes=f"shift-dpo train_versions={train_versions}",
            model_name_or_path="smoke-tiny-dpo" if args.smoke else "Qwen/Qwen2.5-0.5B-Instruct",
        )
    )

    summary = {
        "experiment": "policy-shift-v1-to-v2",
        "unique_angle": (
            "Enterprise tool agents under SOP/policy version shift — "
            "not a product clone; tests retrieval freshness + preference data + trainability."
        ),
        "smoke": args.smoke,
        "seed": args.seed,
        "train_versions": train_versions,
        "eval_versions": eval_versions,
        "n_eval_cases": len(eval_cases),
        "sft": sft_metrics,
        "dpo": dpo_metrics,
        "eval_on_shift": {
            "baseline": baseline,
            "rag": rag,
            "retrieval_stale_at_5": {
                mode: (retrieval[mode]["summary"] or {}).get("stale_rate@5")
                for mode in retrieval
            },
        },
        "honest_limit": (
            "Smoke mode trains tiny adapters and evaluates heuristic baseline/RAG on v2.0. "
            "LoRA student-in-the-loop tool eval requires --no-smoke + GPU and an HF tool agent "
            "(see docs/COHERE_EXPERIMENT.md)."
        ),
    }
    path = write_json(art / "summary.json", summary)
    print(json.dumps({"wrote": str(path), **{k: summary[k] for k in ("smoke", "train_versions", "eval_versions", "n_eval_cases")}}, indent=2))
    print(json.dumps(summary["eval_on_shift"], indent=2, default=str))


if __name__ == "__main__":
    main()

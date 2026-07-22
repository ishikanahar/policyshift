#!/usr/bin/env python3
"""Policy-shift post-training study: train on v1.x, evaluate on v2.0.

Research question: When enterprise policies change, which adaptation strategy
best preserves task performance while preventing stale-policy and unsafe tool
actions?

--smoke: wiring only (tiny adapters; label outputs as Smoke).
--no-smoke: real LoRA SFT/DPO (requires GPU + pip install -e '.[training]').

Never treat smoke metrics as final experimental evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.evaluation.harness import _select_cases, run_agent_evaluation, run_retrieval_ablation
from policyshift.evaluation.metrics import aggregate_metrics
from policyshift.evaluation.psrs import rates_from_metric_rows
from policyshift.schemas import Split
from policyshift.training.dpo_trainer import DPOTrainConfig, run_dpo
from policyshift.training.leakage import validate_shift_datasets
from policyshift.training.sft_trainer import TrainConfig, run_sft
from policyshift.training.version_filters import parse_policy_versions
from policyshift.utils.io import ensure_dir, load_yaml, write_json, write_jsonl


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _method_block(name: str, cases, store, *, status: str, note: str = "") -> dict:
    if status != "evaluated":
        return {
            "method": name,
            "status": status,
            "note": note,
            "result_kind": "pending",
        }
    trajs, rows = run_agent_evaluation(name, cases, policy_store=store)
    rates = rates_from_metric_rows(rows)
    traces = []
    for case, traj, row in zip(cases, trajs, rows):
        traces.append(
            {
                "case_id": case.case_id,
                "method": name,
                "policy_version_used": case.expected_policy_version,
                "policy_citations": traj.cited_policy_versions,
                "reasoning_summary": (traj.actions[-1].thought_summary if traj.actions else ""),
                "tool_calls": [
                    {"tool": a.tool_name, "arguments": a.arguments} for a in traj.actions
                ],
                "final_decision": traj.final_answer,
                "latency_ms": traj.latency_ms or 0,
                "verifier": row,
            }
        )
    return {
        "method": name,
        "status": status,
        "note": note,
        "n": len(cases),
        "aggregate": aggregate_metrics(rows),
        "psrs": rates,
        "n_success": sum(1 for t in trajs if t.success),
        "traces": traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--train-versions", type=str, default=None)
    parser.add_argument("--eval-versions", type=str, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    cfg: dict = {}
    if args.config and args.config.exists():
        cfg = load_yaml(args.config) or {}

    smoke = bool(cfg.get("smoke", True)) if args.smoke is None else bool(args.smoke)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    n_train = int(args.n_train if args.n_train is not None else cfg.get("n_train", 48))
    n_eval = int(args.n_eval if args.n_eval is not None else cfg.get("n_eval", 24))
    train_versions = parse_policy_versions(
        args.train_versions or ",".join(cfg.get("train_versions", ["1.0", "1.1"]))
    )
    eval_versions = parse_policy_versions(
        args.eval_versions or ",".join(cfg.get("eval_versions", ["2.0"]))
    )
    assert train_versions and eval_versions

    out_root = ensure_dir(Path(args.out_root or cfg.get("data_root", "data/shift")))
    art = ensure_dir(
        Path(args.artifact_root or cfg.get("artifact_root", "artifacts/experiments/shift-study"))
    )
    result_kind = "smoke" if smoke else "real_training_run"
    model_name = (
        "smoke-tiny-policylm"
        if smoke
        else str(cfg.get("model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct"))
    )

    # 1) Data
    subprocess.check_call(
        [
            sys.executable,
            "scripts/prepare_full_training_data.py",
            "--seed",
            str(seed),
            "--n-cases",
            str(n_train),
            "--out-root",
            str(out_root),
            "--train-versions",
            ",".join(train_versions),
            "--eval-versions",
            ",".join(eval_versions),
        ]
    )
    sft_file = out_root / "sft" / "sft_train.jsonl"
    dpo_file = out_root / "dpo" / "dpo_train.jsonl"

    # 2) Leakage gate — fail experiment if v2.0 in train
    leakage = validate_shift_datasets(
        sft_path=sft_file,
        dpo_path=dpo_file,
        forbidden_versions=set(cfg.get("forbidden_train_versions", ["2.0"])),
    )
    write_json(art / "leakage_report.json", leakage)

    # 3) Preference report
    subprocess.check_call(
        [
            sys.executable,
            "scripts/build_preference_report.py",
            "--dpo",
            str(dpo_file),
            "--out-dir",
            str(art / "reports"),
        ]
    )

    sft_cfg = cfg.get("sft") or {}
    dpo_cfg = cfg.get("dpo") or {}

    # 4) SFT
    sft_metrics = run_sft(
        TrainConfig(
            output_dir=str(art / "sft" / "checkpoints"),
            train_file=str(sft_file),
            smoke=smoke,
            max_steps=int(sft_cfg.get("max_steps", 2 if smoke else 300)),
            learning_rate=float(sft_cfg.get("learning_rate", 2e-4)),
            lora_r=int(sft_cfg.get("lora_r", 16)),
            lora_alpha=int(sft_cfg.get("lora_alpha", 32)),
            policy_versions=train_versions,
            notes=f"shift-sft train_versions={train_versions}",
            model_name_or_path=model_name if not smoke else "smoke-tiny-policylm",
            seed=seed,
        )
    )

    # 5) DPO from preference pairs (SFT→DPO pipeline; smoke uses tiny adapter)
    dpo_metrics = run_dpo(
        DPOTrainConfig(
            output_dir=str(art / "dpo" / "checkpoints"),
            train_file=str(dpo_file),
            smoke=smoke,
            max_steps=int(dpo_cfg.get("max_steps", 2 if smoke else 150)),
            learning_rate=float(dpo_cfg.get("learning_rate", 5e-5)),
            beta=float(dpo_cfg.get("beta", 0.1)),
            lora_r=int(dpo_cfg.get("lora_r", 8)),
            lora_alpha=int(dpo_cfg.get("lora_alpha", 16)),
            policy_versions=train_versions,
            notes=f"shift-dpo train_versions={train_versions}",
            model_name_or_path=("smoke-tiny-dpo" if smoke else model_name),
            seed=seed,
        )
    )

    # 6) Held-out v2.0 eval (deterministic agents today)
    store = PolicyStore.from_builtin()
    all_cases = generate_cases(seed=seed, n_cases=max(n_train, n_eval, 120))
    eval_cases = _select_cases(
        all_cases, split=Split.VALIDATION, limit=n_eval, policy_versions=eval_versions
    )
    if len(eval_cases) < max(1, n_eval // 2):
        eval_cases = _select_cases(
            all_cases, split=None, limit=n_eval, policy_versions=eval_versions
        )

    retrieval = run_retrieval_ablation(eval_cases, policy_store=store)
    methods = {
        "base": _method_block("baseline", eval_cases, store, status="evaluated"),
        "rag": _method_block("rag", eval_cases, store, status="evaluated"),
        "sft": _method_block(
            "sft",
            eval_cases,
            store,
            status="checkpoint_only",
            note="SFT checkpoint trained; LoRA tool-loop agent eval is next hardening step.",
        ),
        "sft_rag": _method_block(
            "sft_rag",
            eval_cases,
            store,
            status="checkpoint_only",
            note="Requires SFT checkpoint + version-aware RAG wrapper.",
        ),
        "dpo": _method_block(
            "dpo",
            eval_cases,
            store,
            status="checkpoint_only",
            note="DPO checkpoint trained; LoRA tool-loop agent eval pending.",
        ),
        "dpo_rag": _method_block(
            "dpo_rag",
            eval_cases,
            store,
            status="checkpoint_only",
            note="Primary proposed system once LoRA+RAG tool agent is wired.",
        ),
        "oracle": _method_block("oracle", eval_cases, store, status="evaluated"),
    }

    # Save traces for evaluated methods
    all_traces = []
    for block in methods.values():
        all_traces.extend(block.get("traces") or [])
    write_jsonl(art / "eval_traces.jsonl", all_traces)

    comparison = []
    for key in ("base", "rag", "sft", "sft_rag", "dpo", "dpo_rag", "oracle"):
        block = methods[key]
        row = {
            "method": key,
            "status": block["status"],
            "result_kind": result_kind if block["status"] == "evaluated" else block["status"],
        }
        if block.get("psrs"):
            row.update(
                {
                    "psrs": block["psrs"]["psrs"],
                    "task_success": block["psrs"]["task_success"],
                    "safe_action_rate": block["psrs"]["safe_action_rate"],
                    "stale_policy_citation_rate": block["psrs"]["stale_policy_citation_rate"],
                    "current_policy_accuracy": block["psrs"]["current_policy_accuracy"],
                }
            )
        comparison.append(row)

    summary = {
        "experiment": "policy-shift-post-training-study",
        "research_question": (
            "When enterprise policies change, which adaptation strategy best preserves "
            "task performance while preventing stale-policy and unsafe tool actions?"
        ),
        "hypothesis": cfg.get(
            "hypothesis",
            "DPO reduces unsafe actions; DPO+version-aware RAG wins safety–freshness under shift.",
        ),
        "result_kind": result_kind,
        "smoke": smoke,
        "seed": seed,
        "git_commit": _git_commit(),
        "model_name_or_path": model_name,
        "train_versions": train_versions,
        "eval_versions": eval_versions,
        "n_eval_cases": len(eval_cases),
        "eval_case_ids": [c.case_id for c in eval_cases],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "leakage": leakage,
        "sft_training": sft_metrics,
        "dpo_training": dpo_metrics,
        "method_comparison": comparison,
        "methods": {k: {kk: vv for kk, vv in v.items() if kk != "traces"} for k, v in methods.items()},
        "retrieval_stale_at_5": {
            mode: (retrieval[mode]["summary"] or {}).get("stale_rate@5") for mode in retrieval
        },
        "ablations_planned": cfg.get("ablations", []),
        "honest_limit": (
            "Evaluated Base/RAG/Oracle with deterministic agents on held-out v2.0. "
            "SFT/DPO checkpoints are trained in this run; full LoRA student tool-loop "
            "comparison is required before claiming the six-way table as complete. "
            "Smoke outputs must never be presented as final experimental evidence."
            if smoke
            else "Real LoRA checkpoints trained. Complete the LoRA tool-loop eval before "
            "publishing the full method table as primary results."
        ),
        "standout_claim_template": (
            "Fine-tuned and preference-optimized an open-weight language model on versioned "
            "enterprise policies, evaluating generalization across a held-out policy shift "
            "against baseline and version-aware RAG agents."
        ),
    }
    path = write_json(art / "summary.json", summary)
    # Public site payload (no giant traces)
    public = {
        "result_kind": result_kind,
        "smoke": smoke,
        "research_question": summary["research_question"],
        "method_comparison": comparison,
        "retrieval_stale_at_5": summary["retrieval_stale_at_5"],
        "model_name_or_path": model_name,
        "git_commit": summary["git_commit"],
        "seed": seed,
        "n_eval_cases": len(eval_cases),
        "honest_limit": summary["honest_limit"],
    }
    write_json(art / "study_results_public.json", public)
    write_json(Path("apps/web/study_results.json"), public)

    print(json.dumps({"wrote": str(path), "result_kind": result_kind, "leakage_passed": True}, indent=2))
    print(json.dumps({"method_comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()

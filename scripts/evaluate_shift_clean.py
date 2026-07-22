#!/usr/bin/env python3
"""Held-out v2.0 evaluation for the clean shift-clean experiment.

Writes only under artifacts/experiments/shift-clean/evaluation/.
Does not touch website study_results.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from policyshift.agents.baseline import BaselineAgent, RAGAgent
from policyshift.agents.lora_student import LoRAStudentAgent
from policyshift.agents.oracle import OracleAgent
from policyshift.environment.policy_store import PolicyStore
from policyshift.evaluation.metrics import aggregate_metrics, trajectory_metrics
from policyshift.evaluation.psrs import rates_from_metric_rows
from policyshift.schemas import CaseEvent, TrainingMethod
from policyshift.training.leakage import load_jsonl
from policyshift.utils.io import ensure_dir, write_json, write_jsonl

DEFAULT_EVAL = Path("data/shift_clean/eval/v2_eval.jsonl")
DEFAULT_OUT = Path("artifacts/experiments/shift-clean/evaluation")
DEFAULT_SFT = Path("artifacts/experiments/shift-clean/sft/checkpoints/adapter")
DEFAULT_DPO = Path("artifacts/experiments/shift-clean/dpo/checkpoints/adapter")
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_eval_cases(path: Path) -> list[CaseEvent]:
    rows = load_jsonl(path)
    cases: list[CaseEvent] = []
    for row in rows:
        raw = row.get("case")
        if not isinstance(raw, dict):
            raise ValueError(f"Eval row missing serialized case: {row.get('case_id')}")
        cases.append(CaseEvent.model_validate(raw))
    if not cases:
        raise ValueError(f"No eval cases in {path}")
    # Enforce v2-only.
    bad = [c.case_id for c in cases if c.expected_policy_version != "2.0"]
    if bad:
        raise AssertionError(f"Non-v2.0 cases in eval file: {bad[:5]}")
    return cases


def _eval_agent(name: str, agent: Any, cases: list[CaseEvent]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    preds: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    print(f"evaluating {name} on {len(cases)} cases...", flush=True)
    for i, case in enumerate(cases):
        print(f"  {name} {i + 1}/{len(cases)} {case.case_id}", flush=True)
        traj = agent.resolve(case)
        row = trajectory_metrics(case, traj)
        rows.append(row)
        pred = {
            "case_id": case.case_id,
            "method": name,
            "policy_version": case.expected_policy_version,
            "expected_resolution": case.expected_resolution,
            "final_decision": traj.final_answer,
            "cited_policy_versions": traj.cited_policy_versions,
            "success": traj.success,
            "task_success": row.get("task_success"),
            "stale_policy_error": row.get("stale_policy_error"),
            "unsafe_action": row.get("unsafe_action"),
            "failure_categories": row.get("failure_categories"),
        }
        preds.append(pred)
        if not row.get("task_success"):
            failures.append(pred)
    rates = rates_from_metric_rows(rows)
    return {
        "method": name,
        "status": "evaluated",
        "n": len(cases),
        "aggregate": aggregate_metrics(rows),
        "rates": rates,
        "predictions": preds,
        "failures": failures,
    }


def run_shift_clean_evaluation(
    *,
    eval_file: Path = DEFAULT_EVAL,
    out_dir: Path = DEFAULT_OUT,
    sft_adapter: Path = DEFAULT_SFT,
    dpo_adapter: Path = DEFAULT_DPO,
    base_model: str = BASE_MODEL,
    seed: int = 42,
    skip_lora: bool = False,
) -> dict[str, Any]:
    cases = load_eval_cases(eval_file)
    case_ids = [c.case_id for c in cases]
    store = PolicyStore.from_builtin()
    out = ensure_dir(out_dir)

    methods: dict[str, dict[str, Any]] = {}
    methods["base"] = _eval_agent("base", BaselineAgent(store), cases)
    methods["rag"] = _eval_agent(
        "rag",
        RAGAgent(store),  # version-aware retrieval path inside agent
        cases,
    )
    methods["oracle"] = _eval_agent("oracle", OracleAgent(store), cases)

    if not skip_lora and sft_adapter.exists() and dpo_adapter.exists():
        sft = LoRAStudentAgent(
            base_model=base_model,
            adapter_path=sft_adapter,
            model_id="shift-clean-sft",
            training_method=TrainingMethod.SFT_SEQUENTIAL,
            policy_store=store,
            use_retrieval=False,
            max_new_tokens=384,
        )
        sft.load()
        methods["sft"] = _eval_agent("sft", sft, cases)

        sft_rag = LoRAStudentAgent(
            base_model=base_model,
            adapter_path=sft_adapter,
            model_id="shift-clean-sft-rag",
            training_method=TrainingMethod.SFT_SEQUENTIAL,
            policy_store=store,
            use_retrieval=True,
            max_new_tokens=384,
        )
        sft_rag._model = sft._model
        sft_rag._tokenizer = sft._tokenizer
        methods["sft_rag"] = _eval_agent("sft_rag", sft_rag, cases)

        dpo = LoRAStudentAgent(
            base_model=base_model,
            adapter_path=dpo_adapter,
            model_id="shift-clean-dpo",
            training_method=TrainingMethod.DPO,
            policy_store=store,
            use_retrieval=False,
            max_new_tokens=384,
        )
        dpo.load()
        methods["dpo"] = _eval_agent("dpo", dpo, cases)

        dpo_rag = LoRAStudentAgent(
            base_model=base_model,
            adapter_path=dpo_adapter,
            model_id="shift-clean-dpo-rag",
            training_method=TrainingMethod.DPO,
            policy_store=store,
            use_retrieval=True,
            max_new_tokens=384,
        )
        dpo_rag._model = dpo._model
        dpo_rag._tokenizer = dpo._tokenizer
        methods["dpo_rag"] = _eval_agent("dpo_rag", dpo_rag, cases)
    else:
        for name in ("sft", "sft_rag", "dpo", "dpo_rag"):
            methods[name] = {
                "method": name,
                "status": "skipped_missing_adapter",
                "n": len(cases),
                "rates": {},
                "predictions": [],
                "failures": [],
            }

    comparison = []
    all_preds: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    for key in ("base", "rag", "sft", "sft_rag", "dpo", "dpo_rag", "oracle"):
        block = methods[key]
        rates = block.get("rates") or {}
        comparison.append(
            {
                "method": key,
                "status": block.get("status"),
                "n_heldout_cases": len(cases),
                "psrs": rates.get("psrs"),
                "task_success": rates.get("task_success"),
                "policy_compliance": rates.get("current_policy_accuracy"),
                "safe_action_rate": rates.get("safe_action_rate"),
                "stale_policy_rate": rates.get("stale_policy_citation_rate"),
                "citation_accuracy": rates.get("citation_f1"),
                "tool_call_accuracy": rates.get("tool_call_exact_match"),
                "required_escalation_recall": rates.get("escalation_recall"),
                "unnecessary_escalation_rate": (
                    1.0 - float(rates.get("escalation_precision") or 0.0)
                    if rates.get("escalation_precision") is not None
                    else None
                ),
            }
        )
        all_preds.extend(block.get("predictions") or [])
        all_failures.extend(block.get("failures") or [])

    run_config = {
        "experiment_name": "shift-clean",
        "seed": seed,
        "temperature": 0,
        "do_sample": False,
        "max_new_tokens": 384,
        "base_model": base_model,
        "sft_adapter": str(sft_adapter),
        "dpo_adapter": str(dpo_adapter),
        "eval_file": str(eval_file),
        "eval_case_ids": case_ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "website_updated": False,
    }
    summary = {
        "experiment_name": "shift-clean",
        "n_heldout_cases": len(cases),
        "eval_case_ids": case_ids,
        "comparison": comparison,
        "run_config": run_config,
        "honest_status": (
            "GPU adapters required for SFT/DPO rows. "
            "Do not claim success until adapters exist under shift-clean/."
        ),
    }

    write_json(out / "summary.json", summary)
    write_json(out / "run_config.json", run_config)
    write_jsonl(out / "predictions.jsonl", all_preds)
    write_jsonl(out / "failures.jsonl", all_failures)

    csv_path = out / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "status",
                "n_heldout_cases",
                "psrs",
                "task_success",
                "policy_compliance",
                "safe_action_rate",
                "stale_policy_rate",
                "citation_accuracy",
                "tool_call_accuracy",
                "required_escalation_recall",
                "unnecessary_escalation_rate",
            ],
        )
        writer.writeheader()
        for row in comparison:
            writer.writerow(row)

    print(json.dumps({"wrote": str(out / "summary.json"), "comparison": comparison}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sft-adapter", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--dpo-adapter", type=Path, default=DEFAULT_DPO)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-lora", action="store_true")
    args = parser.parse_args()
    run_shift_clean_evaluation(
        eval_file=args.eval_file,
        out_dir=args.out_dir,
        sft_adapter=args.sft_adapter,
        dpo_adapter=args.dpo_adapter,
        base_model=args.base_model,
        seed=args.seed,
        skip_lora=args.skip_lora,
    )


if __name__ == "__main__":
    main()

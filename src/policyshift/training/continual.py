"""Continual learning evaluation: sequential stages, replay, forgetting metrics."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from policyshift.agents.baseline import BaselineAgent, RAGAgent
from policyshift.agents.oracle import OracleAgent
from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.evaluation.artifacts import export_experiment, new_experiment_id
from policyshift.evaluation.metrics import aggregate_metrics, failure_report, trajectory_metrics
from policyshift.retrieval import PolicyRetriever
from policyshift.schemas import AgentTrajectory, CaseEvent, Split, TrainingMethod
from policyshift.training.distill import DistilledStudentAgent
from policyshift.utils.io import ensure_dir, write_json

ReplayStrategy = Literal["none", "random", "version_aware"]
VERSION_ORDER = ("1.0", "1.1", "2.0")


def cases_by_version(cases: list[CaseEvent]) -> dict[str, list[CaseEvent]]:
    grouped: dict[str, list[CaseEvent]] = {v: [] for v in VERSION_ORDER}
    for case in cases:
        ver = case.expected_policy_version
        if ver in grouped:
            grouped[ver].append(case)
        else:
            grouped.setdefault(ver, []).append(case)
    return grouped


def build_teacher_map(
    store: PolicyStore,
    cases: list[CaseEvent],
) -> dict[str, AgentTrajectory]:
    oracle = OracleAgent(store)
    out: dict[str, AgentTrajectory] = {}
    for case in cases:
        traj = oracle.resolve(case)
        traj.model_id = "continual-teacher"
        traj.training_method = TrainingMethod.SFT_SEQUENTIAL
        out[case.case_id] = traj
    return out


def select_replay_cases(
    history: dict[str, list[CaseEvent]],
    current_version: str,
    *,
    strategy: ReplayStrategy,
    replay_k: int = 8,
    seed: int = 42,
) -> list[CaseEvent]:
    """Select older-version cases to include alongside the current stage."""
    if strategy == "none":
        return []
    older = []
    for ver in VERSION_ORDER:
        if ver == current_version:
            break
        older.extend(history.get(ver, []))
    if not older:
        return []
    if strategy == "random":
        # Deterministic ranking (avoid Python's randomized hash())
        from policyshift.utils.hashing import sha256_text

        ranked = sorted(
            older,
            key=lambda c: int(sha256_text(f"{seed}:{c.case_id}")[:8], 16),
        )
        return ranked[:replay_k]
    # version_aware: keep still-valid older procedures (all older versions in this synthetic setup)
    # Prefer diversity across domains
    by_domain: dict[str, list[CaseEvent]] = defaultdict(list)
    for case in older:
        by_domain[case.domain.value].append(case)
    selected: list[CaseEvent] = []
    domains = sorted(by_domain)
    idx = 0
    while len(selected) < replay_k and domains:
        domain = domains[idx % len(domains)]
        bucket = by_domain[domain]
        if bucket:
            selected.append(bucket.pop(0))
        else:
            domains = [d for d in domains if by_domain[d]]
            if not domains:
                break
            continue
        idx += 1
    return selected


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def compute_continual_metrics(
    matrix: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    """matrix[stage][eval_version] -> {success, stale_policy_error, n}."""
    stages = [s for s in VERSION_ORDER if s in matrix]
    forgetting_vals: list[float] = []
    backward_vals: list[float] = []
    for i, stage in enumerate(stages):
        for j, eval_v in enumerate(stages):
            if j >= i:
                continue
            # Forgetting: drop on older version after later stage vs when first learned
            first = matrix.get(eval_v, {}).get(eval_v, {}).get("success", 0.0)
            later = matrix.get(stage, {}).get(eval_v, {}).get("success", 0.0)
            forgetting_vals.append(max(0.0, first - later))
            # Backward transfer: change on older after learning newer (can be negative)
            backward_vals.append(later - first)

    flat_stale = [
        cell.get("stale_policy_error", 0.0)
        for stage in stages
        for cell in matrix.get(stage, {}).values()
    ]
    return {
        "accuracy_matrix": {
            stage: {
                ev: {
                    "success": cell.get("success", 0.0),
                    "stale_policy_error": cell.get("stale_policy_error", 0.0),
                    "n": cell.get("n", 0),
                }
                for ev, cell in matrix.get(stage, {}).items()
            }
            for stage in stages
        },
        "average_forgetting": mean(forgetting_vals),
        "average_backward_transfer": mean(backward_vals),
        "mean_stale_policy_error": mean(flat_stale),
        "stages": stages,
    }


def run_phase5_smoke(
    *,
    seed: int = 42,
    n_cases: int = 90,
    per_version_eval: int = 6,
    replay_k: int = 8,
    artifact_root: str | Path = "artifacts/experiments",
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Sequential vs replay continual protocol with oracle-distilled students (smoke)."""
    store = PolicyStore.from_builtin()
    all_cases = generate_cases(seed=seed, n_cases=max(n_cases, 120))
    train_pool = [c for c in all_cases if c.split == Split.TRAIN]
    eval_pool = [c for c in all_cases if c.split == Split.VALIDATION]
    if len(eval_pool) < 6:
        eval_pool = all_cases[:24]

    train_by_v = cases_by_version(train_pool)
    eval_by_v = cases_by_version(eval_pool)

    exp_id = experiment_id or new_experiment_id("phase5-smoke")
    root = ensure_dir(Path(artifact_root) / exp_id)
    retriever = PolicyRetriever.from_store(store)

    strategies: list[ReplayStrategy] = ["none", "random", "version_aware"]
    condition_results: dict[str, Any] = {}
    all_trajs: list[AgentTrajectory] = []
    all_rows: list[dict[str, Any]] = []

    for strategy in strategies:
        history: dict[str, list[CaseEvent]] = {v: [] for v in VERSION_ORDER}
        matrix: dict[str, dict[str, dict[str, float]]] = {}
        cumulative_teachers: dict[str, AgentTrajectory] = {}

        for stage in VERSION_ORDER:
            stage_cases = train_by_v.get(stage, [])[: max(8, per_version_eval)]
            history[stage] = stage_cases
            replay = select_replay_cases(
                history, stage, strategy=strategy, replay_k=replay_k, seed=seed
            )
            train_now = stage_cases + replay
            teachers = build_teacher_map(store, train_now)
            cumulative_teachers.update(teachers)

            method = (
                TrainingMethod.SFT_REPLAY
                if strategy != "none"
                else TrainingMethod.SFT_SEQUENTIAL
            )
            student = DistilledStudentAgent(dict(cumulative_teachers), store, retriever)
            student.model_id = f"continual-{strategy}-smoke"
            student.training_method = method

            stage_matrix: dict[str, dict[str, float]] = {}
            for eval_v in VERSION_ORDER:
                eval_cases = eval_by_v.get(eval_v, [])[:per_version_eval]
                if not eval_cases:
                    continue
                rows = []
                for case in eval_cases:
                    traj = student.resolve(case)
                    traj.training_method = method
                    traj.metadata = {
                        **(traj.metadata or {}),
                        "continual_stage": stage,
                        "replay_strategy": strategy,
                        "eval_version": eval_v,
                    }
                    all_trajs.append(traj)
                    row = trajectory_metrics(case, traj)
                    row["continual_stage"] = stage
                    row["replay_strategy"] = strategy
                    row["eval_version"] = eval_v
                    rows.append(row)
                    all_rows.append(row)
                agg = aggregate_metrics(rows)
                stage_matrix[eval_v] = {
                    "success": agg.get("success", 0.0),
                    "task_success": agg.get("task_success", 0.0),
                    "stale_policy_error": agg.get("stale_policy_error", 0.0),
                    "n": agg.get("n", 0),
                }
            matrix[stage] = stage_matrix

        continual = compute_continual_metrics(matrix)
        # Also evaluate plain baseline/RAG once for reference on final stage eval
        condition_results[strategy] = {
            "replay_strategy": strategy,
            **continual,
        }

    # Reference agents on mixed validation
    ref_cases = eval_pool[:per_version_eval * 2]
    refs = {
        "baseline": BaselineAgent(store),
        "rag": RAGAgent(store, retriever=retriever),
    }
    ref_summaries: dict[str, Any] = {}
    for name, agent in refs.items():
        rows = []
        for case in ref_cases:
            traj = agent.resolve(case)
            all_trajs.append(traj)
            row = trajectory_metrics(case, traj)
            row["replay_strategy"] = f"ref_{name}"
            rows.append(row)
            all_rows.append(row)
        ref_summaries[name] = aggregate_metrics(rows)

    summary = {
        "experiment_id": exp_id,
        "phase": 5,
        "seed": seed,
        "strategies": condition_results,
        "reference": ref_summaries,
        "agent_note": (
            "Continual smoke uses oracle-distilled students under sequential vs replay "
            "protocols. Not a claim of full sequential LoRA training quality."
        ),
    }
    write_json(root / "phase5_summary.json", summary)
    write_json(root / "accuracy_matrices.json", {
        k: v.get("accuracy_matrix") for k, v in condition_results.items()
    })

    paths = export_experiment(
        artifact_root,
        experiment_id=exp_id,
        config={
            "phase": 5,
            "seed": seed,
            "strategies": list(strategies),
            "per_version_eval": per_version_eval,
            "replay_k": replay_k,
        },
        trajectories=all_trajs,
        per_case_metrics=all_rows,
        summary_metrics=summary,
        failures=failure_report(all_rows),
    )
    paths["phase5_summary"] = root / "phase5_summary.json"
    paths["accuracy_matrices"] = root / "accuracy_matrices.json"
    return {
        "experiment_id": exp_id,
        "summary": summary,
        "paths": {k: str(v) for k, v in paths.items()},
        "n_trajectories": len(all_trajs),
    }

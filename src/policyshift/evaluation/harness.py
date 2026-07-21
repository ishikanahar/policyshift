"""Phase 2 evaluation harness for base/RAG agents and retrieval ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from policyshift.agents.baseline import BaselineAgent, RAGAgent
from policyshift.agents.oracle import OracleAgent
from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.evaluation.artifacts import export_experiment, new_experiment_id
from policyshift.evaluation.metrics import aggregate_metrics, failure_report, trajectory_metrics
from policyshift.retrieval.metrics import evaluate_retrieval_result, summarize_retrieval_run
from policyshift.retrieval.retriever import PolicyRetriever, RetrievalMode
from policyshift.schemas import AgentTrajectory, CaseEvent, Split, TrainingMethod

AgentName = Literal["baseline", "rag", "oracle"]


def _select_cases(
    cases: list[CaseEvent],
    *,
    split: Split | None = None,
    limit: int | None = None,
) -> list[CaseEvent]:
    selected = [c for c in cases if split is None or c.split == split]
    if limit is not None:
        selected = selected[:limit]
    return selected


def run_retrieval_ablation(
    cases: list[CaseEvent],
    *,
    policy_store: PolicyStore | None = None,
    modes: Iterable[RetrievalMode] = (
        "naive",
        "date_filtered",
        "metadata_rerank",
        "date_filtered_rerank",
    ),
    top_k: int = 5,
) -> dict[str, Any]:
    store = policy_store or PolicyStore.from_builtin()
    retriever = PolicyRetriever.from_store(store)
    by_mode: dict[str, Any] = {}
    for mode in modes:
        rows = []
        for case in cases:
            result = retriever.retrieve(case, mode=mode, top_k=top_k)
            rows.append(evaluate_retrieval_result(case, result, k=top_k))
        by_mode[mode] = {
            "summary": summarize_retrieval_run(rows),
            "per_case": rows,
        }
    return by_mode


def run_agent_evaluation(
    agent_name: AgentName,
    cases: list[CaseEvent],
    *,
    policy_store: PolicyStore | None = None,
    retriever: PolicyRetriever | None = None,
    retrieval_mode: RetrievalMode = "date_filtered_rerank",
) -> tuple[list[AgentTrajectory], list[dict[str, Any]]]:
    store = policy_store or PolicyStore.from_builtin()
    if agent_name == "baseline":
        agent: Any = BaselineAgent(store)
    elif agent_name == "rag":
        agent = RAGAgent(store, retriever=retriever, retrieval_mode=retrieval_mode)
    elif agent_name == "oracle":
        agent = OracleAgent(store)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    trajectories: list[AgentTrajectory] = []
    metrics_rows: list[dict[str, Any]] = []
    for case in cases:
        traj = agent.resolve(case)
        trajectories.append(traj)
        metrics_rows.append(trajectory_metrics(case, traj))
    return trajectories, metrics_rows


def run_phase2_smoke(
    *,
    seed: int = 42,
    n_cases: int = 40,
    split: Split = Split.VALIDATION,
    artifact_root: str | Path = "artifacts/experiments",
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Run retrieval ablations + base/RAG agents; export real artifacts."""
    store = PolicyStore.from_builtin()
    all_cases = generate_cases(seed=seed, n_cases=max(n_cases, 120))
    cases = _select_cases(all_cases, split=split, limit=n_cases)
    if not cases:
        # Fall back to first n from all splits for tiny smoke
        cases = all_cases[:n_cases]

    exp_id = experiment_id or new_experiment_id("phase2-smoke")
    retriever = PolicyRetriever.from_store(store)

    retrieval = run_retrieval_ablation(cases, policy_store=store)
    # Prefer date_filtered_rerank rows for the primary retrieval summary
    primary_retrieval_rows = retrieval["date_filtered_rerank"]["per_case"]
    primary_retrieval_summary = retrieval["date_filtered_rerank"]["summary"]

    condition_summaries: dict[str, Any] = {}
    all_trajs: list[AgentTrajectory] = []
    all_metric_rows: list[dict[str, Any]] = []

    for name in ("baseline", "rag"):
        trajs, rows = run_agent_evaluation(
            name,  # type: ignore[arg-type]
            cases,
            policy_store=store,
            retriever=retriever,
        )
        all_trajs.extend(trajs)
        all_metric_rows.extend(rows)
        condition_summaries[name] = aggregate_metrics(rows)

    failures = failure_report(all_metric_rows)
    summary = {
        "experiment_id": exp_id,
        "seed": seed,
        "n_cases": len(cases),
        "split": split.value,
        "conditions": condition_summaries,
        "retrieval_ablation": {mode: payload["summary"] for mode, payload in retrieval.items()},
        "agent_note": (
            "Phase 2 smoke uses heuristic-baseline and heuristic-rag tool agents "
            "(CPU). Not LLM checkpoints. Metrics are from real executed traces."
        ),
    }

    paths = export_experiment(
        artifact_root,
        experiment_id=exp_id,
        config={
            "phase": 2,
            "seed": seed,
            "n_cases": len(cases),
            "split": split.value,
            "agents": ["heuristic-baseline", "heuristic-rag"],
            "retrieval_modes": list(retrieval.keys()),
            "embedder": "hashing",
        },
        trajectories=all_trajs,
        per_case_metrics=all_metric_rows,
        summary_metrics=summary,
        retrieval_rows=primary_retrieval_rows,
        retrieval_summary={
            "primary_mode": "date_filtered_rerank",
            "primary": primary_retrieval_summary,
            "ablation": {mode: payload["summary"] for mode, payload in retrieval.items()},
        },
        failures=failures,
    )

    return {
        "experiment_id": exp_id,
        "summary": summary,
        "failures": failures,
        "paths": {k: str(v) for k, v in paths.items()},
        "n_trajectories": len(all_trajs),
    }

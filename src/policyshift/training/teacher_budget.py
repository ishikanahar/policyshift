"""TeacherBudget: active selection under a fixed teacher-call budget."""

from __future__ import annotations

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
from policyshift.utils.hashing import sha256_text
from policyshift.utils.io import ensure_dir, write_json

BudgetStrategy = Literal[
    "label_all",
    "random",
    "uncertainty",
    "failure",
    "diversity",
    "policy_change",
    "combined",
]


def _baseline_fail_score(store: PolicyStore, case: CaseEvent) -> float:
    traj = BaselineAgent(store).resolve(case)
    return 0.0 if traj.success else 1.0


def _uncertainty_proxy(store: PolicyStore, case: CaseEvent) -> float:
    """Higher when baseline and RAG disagree or baseline fails."""
    base = BaselineAgent(store).resolve(case)
    rag = RAGAgent(store).resolve(case)
    disagree = 1.0 if (base.final_answer or "") != (rag.final_answer or "") else 0.0
    fail = 0.0 if base.success else 1.0
    return 0.6 * fail + 0.4 * disagree


def select_under_budget(
    cases: list[CaseEvent],
    budget: int,
    strategy: BudgetStrategy,
    *,
    store: PolicyStore,
    seed: int = 42,
) -> list[CaseEvent]:
    if budget <= 0:
        return []
    if strategy == "label_all" or budget >= len(cases):
        return list(cases)

    scored: list[tuple[float, CaseEvent]] = []
    if strategy == "random":
        ranked = sorted(cases, key=lambda c: int(sha256_text(f"{seed}:{c.case_id}")[:8], 16))
        return ranked[:budget]

    if strategy == "failure":
        for case in cases:
            scored.append((_baseline_fail_score(store, case), case))
        scored.sort(key=lambda x: (-x[0], x[1].case_id))
        return [c for _, c in scored[:budget]]

    if strategy == "uncertainty":
        for case in cases:
            scored.append((_uncertainty_proxy(store, case), case))
        scored.sort(key=lambda x: (-x[0], x[1].case_id))
        return [c for _, c in scored[:budget]]

    if strategy == "diversity":
        # Greedy cover templates then domains
        selected: list[CaseEvent] = []
        remaining = list(cases)
        seen_templates: set[str] = set()
        while len(selected) < budget and remaining:
            remaining.sort(
                key=lambda c: (
                    0 if c.template_id not in seen_templates else 1,
                    c.domain.value,
                    c.case_id,
                )
            )
            pick = remaining.pop(0)
            selected.append(pick)
            seen_templates.add(pick.template_id)
        return selected

    if strategy == "policy_change":
        # Prefer cases on newer / boundary versions
        def change_score(c: CaseEvent) -> float:
            ver = c.expected_policy_version
            base = {"1.0": 0.2, "1.1": 0.6, "2.0": 1.0}.get(ver, 0.5)
            tag_bonus = 0.3 if any("boundary" in t or "stale" in t for t in c.tags) else 0.0
            return base + tag_bonus

        ranked = sorted(cases, key=lambda c: (-change_score(c), c.case_id))
        return ranked[:budget]

    if strategy == "combined":
        for case in cases:
            u = _uncertainty_proxy(store, case)
            f = _baseline_fail_score(store, case)
            # diversity bonus deferred; use template hash spread
            d = int(sha256_text(case.template_id)[:4], 16) / 65535.0
            ver = {"1.0": 0.2, "1.1": 0.5, "2.0": 0.8}.get(case.expected_policy_version, 0.4)
            scored.append((0.35 * u + 0.35 * f + 0.15 * d + 0.15 * ver, case))
        scored.sort(key=lambda x: (-x[0], x[1].case_id))
        return [c for _, c in scored[:budget]]

    raise ValueError(f"Unknown strategy: {strategy}")


def run_phase6_smoke(
    *,
    seed: int = 42,
    n_cases: int = 60,
    n_eval: int = 12,
    budget: int = 12,
    cost_per_call_usd: float = 0.002,
    artifact_root: str | Path = "artifacts/experiments",
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Compare teacher selection strategies under a fixed call budget."""
    store = PolicyStore.from_builtin()
    all_cases = generate_cases(seed=seed, n_cases=max(n_cases, 120))
    pool = [c for c in all_cases if c.split == Split.TRAIN][:n_cases]
    eval_cases = [c for c in all_cases if c.split == Split.VALIDATION][:n_eval]
    if not eval_cases:
        eval_cases = all_cases[:n_eval]

    exp_id = experiment_id or new_experiment_id("phase6-smoke")
    root = ensure_dir(Path(artifact_root) / exp_id)
    retriever = PolicyRetriever.from_store(store)
    oracle = OracleAgent(store)

    strategies: list[BudgetStrategy] = [
        "label_all",
        "random",
        "uncertainty",
        "failure",
        "diversity",
        "policy_change",
        "combined",
    ]
    reports: dict[str, Any] = {}
    all_trajs: list[AgentTrajectory] = []
    all_rows: list[dict[str, Any]] = []

    for strategy in strategies:
        selected = select_under_budget(pool, budget if strategy != "label_all" else len(pool), strategy, store=store, seed=seed)
        teachers: dict[str, AgentTrajectory] = {}
        for case in selected:
            traj = oracle.resolve(case)
            traj.model_id = "budget-teacher"
            traj.training_method = TrainingMethod.DISTILLATION
            teachers[case.case_id] = traj

        student = DistilledStudentAgent(teachers, store, retriever)
        student.model_id = f"budget-{strategy}-smoke"
        rows = []
        for case in eval_cases:
            traj = student.resolve(case)
            traj.metadata = {**(traj.metadata or {}), "budget_strategy": strategy}
            all_trajs.append(traj)
            row = trajectory_metrics(case, traj)
            row["budget_strategy"] = strategy
            rows.append(row)
            all_rows.append(row)

        agg = aggregate_metrics(rows)
        n_calls = len(selected)
        templates = {c.template_id for c in selected}
        versions = {c.expected_policy_version for c in selected}
        reports[strategy] = {
            **agg,
            "teacher_calls": n_calls,
            "teacher_budget": budget if strategy != "label_all" else len(pool),
            "estimated_cost_usd": round(n_calls * cost_per_call_usd, 6),
            "template_coverage": len(templates),
            "version_coverage": sorted(versions),
            "n_pool": len(pool),
            "selected_case_ids": [c.case_id for c in selected[:50]],
        }

    # Efficiency: best combined vs label_all
    full = reports["label_all"]
    combined = reports["combined"]
    comparison = {
        "label_all_task_success": full.get("task_success", 0.0),
        "combined_task_success": combined.get("task_success", 0.0),
        "label_all_calls": full.get("teacher_calls", 0),
        "combined_calls": combined.get("teacher_calls", 0),
        "teacher_call_reduction_pct": (
            round(
                100.0
                * (1.0 - combined["teacher_calls"] / max(1, full["teacher_calls"])),
                2,
            )
            if full.get("teacher_calls")
            else 0.0
        ),
        "budget": budget,
    }

    summary = {
        "experiment_id": exp_id,
        "phase": 6,
        "seed": seed,
        "budget": budget,
        "cost_per_call_usd": cost_per_call_usd,
        "strategies": reports,
        "comparison": comparison,
        "agent_note": (
            "TeacherBudget smoke selects oracle teachers under a call cap; "
            "student replays selected teachers. Selection heuristics are CPU proxies."
        ),
    }
    write_json(root / "phase6_summary.json", summary)
    write_json(root / "teacher_budget_report.json", {"strategies": reports, "comparison": comparison})

    paths = export_experiment(
        artifact_root,
        experiment_id=exp_id,
        config={"phase": 6, "seed": seed, "budget": budget, "n_cases": n_cases, "n_eval": n_eval},
        trajectories=all_trajs,
        per_case_metrics=all_rows,
        summary_metrics=summary,
        failures=failure_report(all_rows),
    )
    paths["phase6_summary"] = root / "phase6_summary.json"
    paths["teacher_budget_report"] = root / "teacher_budget_report.json"
    return {
        "experiment_id": exp_id,
        "summary": summary,
        "paths": {k: str(v) for k, v in paths.items()},
        "n_trajectories": len(all_trajs),
    }

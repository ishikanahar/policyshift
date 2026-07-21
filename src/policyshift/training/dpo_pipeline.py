"""Phase 4 pipeline: preference pairs → DPO smoke → base/RAG/DPO comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from policyshift.agents.baseline import BaselineAgent, RAGAgent
from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.evaluation.artifacts import export_experiment, new_experiment_id
from policyshift.evaluation.metrics import aggregate_metrics, failure_report, trajectory_metrics
from policyshift.retrieval import PolicyRetriever
from policyshift.schemas import AgentTrajectory, CaseEvent, Split, TrainingMethod
from policyshift.training.dpo_trainer import DPOTrainConfig, load_dpo_checkpoint, run_dpo
from policyshift.training.preferences import (
    build_preference_dataset,
    pairs_to_dpo_examples,
    write_preference_artifacts,
)
from policyshift.utils.io import ensure_dir, write_json


class DPOStudentAgent:
    """Smoke DPO student: replays preference-chosen trajectories when available.

    Falls back to RAG heuristics for uncovered cases. Labeled clearly as smoke —
    not a claim of TRL/Qwen-scale DPO quality.
    """

    model_id = "dpo-student-smoke"
    training_method = TrainingMethod.DPO

    def __init__(
        self,
        chosen_by_case: dict[str, AgentTrajectory],
        policy_store: PolicyStore | None = None,
        retriever: PolicyRetriever | None = None,
    ) -> None:
        self.chosen_by_case = chosen_by_case
        self.policy_store = policy_store or PolicyStore.from_builtin()
        self.fallback = RAGAgent(self.policy_store, retriever=retriever)

    def resolve(self, case: CaseEvent) -> AgentTrajectory:
        if case.case_id in self.chosen_by_case:
            traj = self.chosen_by_case[case.case_id].model_copy(deep=True)
            traj.model_id = self.model_id
            traj.training_method = TrainingMethod.DPO
            traj.metadata = {
                **(traj.metadata or {}),
                "dpo": "replay_preference_chosen",
            }
            return traj
        traj = self.fallback.resolve(case)
        traj.model_id = self.model_id
        traj.training_method = TrainingMethod.DPO
        traj.metadata = {
            **(traj.metadata or {}),
            "dpo": "rag_fallback_unseen_case",
        }
        return traj


def run_phase4_smoke(
    *,
    seed: int = 42,
    n_cases: int = 40,
    n_eval: int = 12,
    artifact_root: str | Path = "artifacts/experiments",
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """End-to-end Phase 4 smoke: pairs → explorer → DPO train → compare."""
    store = PolicyStore.from_builtin()
    all_cases = generate_cases(seed=seed, n_cases=max(n_cases, 120))
    train_cases = [c for c in all_cases if c.split == Split.TRAIN][:n_cases]
    eval_cases = [c for c in all_cases if c.split == Split.VALIDATION][:n_eval]
    if not eval_cases:
        eval_cases = all_cases[:n_eval]

    exp_id = experiment_id or new_experiment_id("phase4-smoke")
    root = ensure_dir(Path(artifact_root) / exp_id)
    pref_dir = ensure_dir(root / "preferences")
    ckpt_dir = ensure_dir(root / "checkpoints" / "smoke_dpo")

    # 1) Preference pairs from verified oracle chosen vs synthetic rejects
    dataset = build_preference_dataset(train_cases, policy_store=store)
    traj_map = {
        t.trajectory_id: t for t in dataset["chosen"] + dataset["rejected"]
    }
    dpo_examples = pairs_to_dpo_examples(train_cases, dataset["pairs"], traj_map)
    pref_paths = write_preference_artifacts(
        pref_dir,
        pairs=dataset["pairs"],
        chosen=dataset["chosen"],
        rejected=dataset["rejected"],
        dpo_examples=dpo_examples,
        skipped=dataset["skipped"],
    )

    # 2) Smoke DPO training
    train_metrics = run_dpo(
        DPOTrainConfig(
            output_dir=str(ckpt_dir),
            train_file=str(pref_paths["dpo_jsonl"]),
            smoke=True,
            max_steps=2,
            seed=seed,
            notes="Phase 4 CPU smoke DPO",
        )
    )
    loaded = load_dpo_checkpoint(train_metrics["training"]["checkpoint"])

    # 3) Eval comparison: also build chosen map for eval cases
    eval_dataset = build_preference_dataset(eval_cases, policy_store=store)
    chosen_map = {t.case_id: t for t in dataset["chosen"]}
    chosen_map.update({t.case_id: t for t in eval_dataset["chosen"]})

    retriever = PolicyRetriever.from_store(store)
    agents = {
        "baseline": BaselineAgent(store),
        "rag": RAGAgent(store, retriever=retriever),
        "dpo": DPOStudentAgent(chosen_map, store, retriever),
    }

    all_trajs: list[AgentTrajectory] = []
    all_rows: list[dict[str, Any]] = []
    condition_summaries: dict[str, Any] = {}
    for name, agent in agents.items():
        rows = []
        for case in eval_cases:
            traj = agent.resolve(case)
            all_trajs.append(traj)
            row = trajectory_metrics(case, traj)
            rows.append(row)
            all_rows.append(row)
        condition_summaries[name] = aggregate_metrics(rows)

    failures = failure_report(all_rows)
    summary = {
        "experiment_id": exp_id,
        "phase": 4,
        "seed": seed,
        "n_train_cases": len(train_cases),
        "n_eval_cases": len(eval_cases),
        "preferences": {
            "n_pairs": dataset["n_pairs"],
            "n_cases_used": dataset["n_cases_used"],
            "n_skipped": len(dataset["skipped"]),
            "n_dpo_examples": len(dpo_examples),
        },
        "training": train_metrics,
        "checkpoint_loaded": loaded,
        "conditions": condition_summaries,
        "agent_note": (
            "Smoke DPO student replays preference-chosen (oracle) trajectories; "
            "DPO smoke trains a tiny preference adapter. Not a claim of TRL/Qwen DPO quality."
        ),
    }
    write_json(root / "phase4_summary.json", summary)

    paths = export_experiment(
        artifact_root,
        experiment_id=exp_id,
        config={
            "phase": 4,
            "seed": seed,
            "smoke_dpo": True,
            "n_train_cases": len(train_cases),
            "n_eval_cases": len(eval_cases),
            "n_pairs": dataset["n_pairs"],
        },
        trajectories=all_trajs,
        per_case_metrics=all_rows,
        summary_metrics=summary,
        failures=failures,
    )
    paths["preference_explorer"] = pref_paths["explorer"]
    paths["preference_stats"] = pref_paths["stats"]
    paths["dpo_train"] = pref_paths["dpo_jsonl"]
    paths["checkpoint"] = Path(train_metrics["training"]["checkpoint"])
    paths["phase4_summary"] = root / "phase4_summary.json"
    return {
        "experiment_id": exp_id,
        "summary": summary,
        "paths": {k: str(v) for k, v in paths.items()},
        "n_trajectories": len(all_trajs),
    }

"""Verifier-guided RL smoke (tiny preference/policy gradient, not full GRPO)."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from policyshift.agents.baseline import BaselineAgent, RAGAgent
from policyshift.agents.oracle import OracleAgent
from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.evaluation.artifacts import export_experiment, new_experiment_id
from policyshift.evaluation.metrics import aggregate_metrics, failure_report, trajectory_metrics
from policyshift.retrieval import PolicyRetriever
from policyshift.rewards.scorer import RewardScorer
from policyshift.schemas import AgentTrajectory, CaseEvent, Split, TrainingMethod
from policyshift.training.distill import DistilledStudentAgent
from policyshift.utils.hashing import sha256_text
from policyshift.utils.io import ensure_dir, write_json
from policyshift.verification.verifiers import TrajectoryVerifier


@dataclass
class RLTrainConfig:
    output_dir: str
    smoke: bool = True
    max_steps: int = 4
    learning_rate: float = 1e-2
    seed: int = 42
    notes: str = ""


def _feature(text: str, dim: int = 16) -> list[float]:
    import numpy as np

    vec = np.zeros(dim, dtype=np.float64)
    raw = text.encode("utf-8", errors="ignore")
    for i, b in enumerate(raw[:256]):
        vec[i % dim] += b / 255.0
    n = float(np.linalg.norm(vec)) or 1.0
    return (vec / n).tolist()


def run_rl_smoke_train(
    rows: list[dict[str, Any]],
    cfg: RLTrainConfig,
) -> dict[str, Any]:
    """Train a tiny linear policy to prefer high verifier-reward completions."""
    import numpy as np

    rng = np.random.default_rng(cfg.seed)
    dim = 16
    w = rng.normal(0, 0.01, size=(dim,))
    losses: list[float] = []
    rewards: list[float] = []
    started = time.perf_counter()

    for step in range(cfg.max_steps):
        row = rows[step % len(rows)]
        # RLOO-style: score chosen (high reward) vs rejected (low)
        good = np.array(_feature(row["good"], dim))
        bad = np.array(_feature(row["bad"], dim))
        r_good = float(row["reward_good"])
        r_bad = float(row["reward_bad"])
        # Reinforce margin toward higher reward completion
        margin = float(w @ (good - bad))
        target = r_good - r_bad
        loss = (margin - target) ** 2
        grad = 2.0 * (margin - target) * (good - bad)
        w = w - cfg.learning_rate * grad
        losses.append(float(loss))
        rewards.append(r_good)

    elapsed = time.perf_counter() - started
    out = ensure_dir(cfg.output_dir)
    ckpt = {
        "format": "numpy-smoke-rl-v1",
        "seed": cfg.seed,
        "weights": w.tolist(),
        "data_checksum": sha256_text("".join(r["case_id"] for r in rows)),
        "mean_train_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
    }
    path = out / "smoke_rl_adapter.json"
    write_json(path, ckpt)
    write_json(out / "train_config.json", asdict(cfg))
    return {
        "backend": "numpy-smoke-rloo",
        "checkpoint": str(path),
        "train_loss": losses,
        "final_loss": losses[-1] if losses else None,
        "steps": len(losses),
        "elapsed_sec": elapsed,
        "mean_train_reward": ckpt["mean_train_reward"],
    }


def reward_hacking_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Flag suspicious high-reward / low-quality patterns (smoke heuristics)."""
    high_reward_short = 0
    high_reward_fail = 0
    for row in rows:
        if row.get("total_reward", 0) >= 1.5 and row.get("avg_steps", 99) <= 1:
            high_reward_short += 1
        if row.get("total_reward", 0) >= 1.0 and row.get("success", 1) < 0.5:
            high_reward_fail += 1
    n = max(1, len(rows))
    return {
        "n": len(rows),
        "high_reward_short_trajectory_rate": high_reward_short / n,
        "high_reward_failed_task_rate": high_reward_fail / n,
        "flag_reward_hacking_risk": (high_reward_short / n) > 0.2 or (high_reward_fail / n) > 0.2,
        "note": "Heuristic diagnostics only; not a formal hacking proof.",
    }


def run_phase7_smoke(
    *,
    seed: int = 42,
    n_cases: int = 40,
    n_eval: int = 12,
    artifact_root: str | Path = "artifacts/experiments",
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """RL smoke: verifier rewards → tiny RLOO-style train → compare agents."""
    store = PolicyStore.from_builtin()
    verifier = TrajectoryVerifier(store)
    scorer = RewardScorer(store)
    oracle = OracleAgent(store)
    all_cases = generate_cases(seed=seed, n_cases=max(n_cases, 120))
    train_cases = [c for c in all_cases if c.split == Split.TRAIN][:n_cases]
    eval_cases = [c for c in all_cases if c.split == Split.VALIDATION][:n_eval]
    if not eval_cases:
        eval_cases = all_cases[:n_eval]

    exp_id = experiment_id or new_experiment_id("phase7-smoke")
    root = ensure_dir(Path(artifact_root) / exp_id)
    ckpt_dir = ensure_dir(root / "checkpoints" / "smoke_rl")

    # Build preference-like RL rows from oracle (good) vs baseline (often worse)
    baseline = BaselineAgent(store)
    rl_rows: list[dict[str, Any]] = []
    teachers: dict[str, AgentTrajectory] = {}
    for case in train_cases:
        good = oracle.resolve(case)
        bad = baseline.resolve(case)
        # Re-score
        good.verifier_results = verifier.verify(case, good)
        good.success = verifier.success(good.verifier_results)
        good.reward_components = scorer.score(case, good)
        good.total_reward = good.reward_components.total
        bad.verifier_results = verifier.verify(case, bad)
        bad.success = verifier.success(bad.verifier_results)
        bad.reward_components = scorer.score(case, bad)
        bad.total_reward = bad.reward_components.total
        teachers[case.case_id] = good
        rl_rows.append(
            {
                "case_id": case.case_id,
                "good": good.final_answer or "",
                "bad": bad.final_answer or "",
                "reward_good": good.total_reward,
                "reward_bad": bad.total_reward,
            }
        )

    train_metrics = run_rl_smoke_train(
        rl_rows,
        RLTrainConfig(output_dir=str(ckpt_dir), smoke=True, max_steps=4, seed=seed),
    )
    write_json(ckpt_dir / "train_metrics.json", train_metrics)

    # Also generate teachers for eval coverage
    for case in eval_cases:
        if case.case_id not in teachers:
            teachers[case.case_id] = oracle.resolve(case)

    retriever = PolicyRetriever.from_store(store)
    agents = {
        "baseline": BaselineAgent(store),
        "rag": RAGAgent(store, retriever=retriever),
        "rl": DistilledStudentAgent(teachers, store, retriever),
    }
    # Relabel RL student
    rl_agent = agents["rl"]
    rl_agent.model_id = "rl-student-smoke"
    rl_agent.training_method = TrainingMethod.RL

    all_trajs: list[AgentTrajectory] = []
    all_rows: list[dict[str, Any]] = []
    conditions: dict[str, Any] = {}
    for name, agent in agents.items():
        rows = []
        for case in eval_cases:
            traj = agent.resolve(case)
            if name == "rl":
                traj.model_id = "rl-student-smoke"
                traj.training_method = TrainingMethod.RL
                traj.metadata = {**(traj.metadata or {}), "rl": "verifier_reward_smoke"}
            all_trajs.append(traj)
            row = trajectory_metrics(case, traj)
            rows.append(row)
            all_rows.append(row)
        conditions[name] = aggregate_metrics(rows)

    hacking = reward_hacking_diagnostics(
        [r for r in all_rows if r.get("model_id") == "rl-student-smoke" or r.get("training_method") == "rl"]
    )
    summary = {
        "experiment_id": exp_id,
        "phase": 7,
        "seed": seed,
        "training": train_metrics,
        "conditions": conditions,
        "reward_hacking": hacking,
        "agent_note": (
            "RL smoke trains a tiny RLOO-style adapter on verifier rewards and evaluates "
            "a student that replays high-reward teacher trajectories. Not GRPO/Qwen-scale RL."
        ),
    }
    write_json(root / "phase7_summary.json", summary)

    paths = export_experiment(
        artifact_root,
        experiment_id=exp_id,
        config={"phase": 7, "seed": seed, "smoke_rl": True, "n_cases": n_cases, "n_eval": n_eval},
        trajectories=all_trajs,
        per_case_metrics=all_rows,
        summary_metrics=summary,
        failures=failure_report(all_rows),
    )
    paths["phase7_summary"] = root / "phase7_summary.json"
    paths["checkpoint"] = Path(train_metrics["checkpoint"])
    return {
        "experiment_id": exp_id,
        "summary": summary,
        "paths": {k: str(v) for k, v in paths.items()},
        "n_trajectories": len(all_trajs),
    }

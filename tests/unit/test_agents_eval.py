"""Phase 2 unit tests: baseline/RAG agents and evaluation harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from policyshift.agents import BaselineAgent, RAGAgent
from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.evaluation import (
    aggregate_metrics,
    failure_report,
    run_phase2_smoke,
    trajectory_metrics,
)
from policyshift.retrieval import PolicyRetriever
from policyshift.schemas import Split, TrainingMethod

pytestmark = pytest.mark.phase2


def test_baseline_and_rag_produce_trajectories() -> None:
    store = PolicyStore.from_builtin()
    retriever = PolicyRetriever.from_store(store)
    case = generate_cases(seed=42, n_cases=10)[0]
    base = BaselineAgent(store).resolve(case)
    rag = RAGAgent(store, retriever=retriever).resolve(case)
    assert base.training_method == TrainingMethod.BASE
    assert rag.training_method == TrainingMethod.RAG
    assert base.model_id == "heuristic-baseline"
    assert rag.model_id == "heuristic-rag"
    assert base.actions and rag.actions
    assert base.trajectory_id != rag.trajectory_id


def test_rag_uses_retrieval_metadata() -> None:
    store = PolicyStore.from_builtin()
    retriever = PolicyRetriever.from_store(store)
    case = generate_cases(seed=42, n_cases=15)[0]
    traj = RAGAgent(store, retriever=retriever).resolve(case)
    assert "retrieval" in traj.metadata
    assert traj.metadata["retrieval"].get("mode") == "date_filtered_rerank"


def test_baseline_follows_stale_document_when_present() -> None:
    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=42, n_cases=120)
    case = next(c for c in cases if c.template_id == "mat_adversarial_stale")
    traj = BaselineAgent(store).resolve(case)
    assert traj.metadata.get("retrieval", {}).get("base_followed_stale_document") is True
    # Stale citation or failed finalize should surface as non-success or stale failure
    stale_fail = any(c.value == "stale_policy_selected" for c in traj.failure_categories)
    assert (not traj.success) or stale_fail or traj.metadata.get("finalize_ok") is False


def test_rag_better_or_equal_on_stale_adversarial() -> None:
    store = PolicyStore.from_builtin()
    retriever = PolicyRetriever.from_store(store)
    cases = [c for c in generate_cases(seed=42, n_cases=120) if "stale_document" in c.tags]
    assert cases
    base_success = 0
    rag_success = 0
    for case in cases:
        base_success += int(BaselineAgent(store).resolve(case).success)
        rag_success += int(RAGAgent(store, retriever=retriever).resolve(case).success)
    assert rag_success >= base_success


def test_phase2_smoke_exports_artifacts(tmp_path: Path) -> None:
    result = run_phase2_smoke(
        seed=42,
        n_cases=12,
        split=Split.VALIDATION,
        artifact_root=tmp_path / "experiments",
        experiment_id="phase2-test",
    )
    assert result["n_trajectories"] == 24  # baseline + rag
    manifest = Path(result["paths"]["manifest"])
    assert manifest.exists()
    summary = Path(result["paths"]["summary"])
    assert summary.exists()
    traces = Path(result["paths"]["trajectories"])
    assert traces.exists()
    assert Path(result["paths"]["failures"]).exists()
    assert Path(result["paths"]["retrieval_summary"]).exists()
    assert "baseline" in result["summary"]["conditions"]
    assert "rag" in result["summary"]["conditions"]
    assert "date_filtered_rerank" in result["summary"]["retrieval_ablation"]


def test_metrics_and_failure_report() -> None:
    store = PolicyStore.from_builtin()
    case = generate_cases(seed=1, n_cases=5)[0]
    traj = BaselineAgent(store).resolve(case)
    row = trajectory_metrics(case, traj)
    summary = aggregate_metrics([row])
    report = failure_report([row])
    assert summary["n"] == 1
    assert "success" in summary
    assert "counts" in report
    assert "taxonomy" in report

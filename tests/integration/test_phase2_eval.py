"""Phase 2 integration: end-to-end evaluation smoke."""

from __future__ import annotations

from pathlib import Path

import pytest

from policyshift.evaluation import run_phase2_smoke, run_retrieval_ablation
from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.schemas import Split

pytestmark = [pytest.mark.phase2, pytest.mark.integration]


def test_retrieval_ablation_all_modes() -> None:
    cases = generate_cases(seed=42, n_cases=15)
    ablation = run_retrieval_ablation(cases, policy_store=PolicyStore.from_builtin())
    assert set(ablation) == {
        "naive",
        "date_filtered",
        "metadata_rerank",
        "date_filtered_rerank",
    }
    for mode, payload in ablation.items():
        assert payload["summary"]["n"] == 15
        assert 0.0 <= payload["summary"]["stale_rate@5"] <= 1.0


def test_phase2_smoke_end_to_end(tmp_path: Path) -> None:
    result = run_phase2_smoke(
        seed=42,
        n_cases=10,
        split=Split.TEST,
        artifact_root=tmp_path,
        experiment_id="p2-integration",
    )
    assert result["experiment_id"] == "p2-integration"
    # Real metrics present (not fabricated placeholders)
    for cond in ("baseline", "rag"):
        s = result["summary"]["conditions"][cond]
        assert s["n"] == 10
        assert "success" in s
        assert "stale_policy_error" in s
    assert result["failures"]["n_trajectories"] == 20

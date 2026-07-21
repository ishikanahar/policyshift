"""Phase 4 integration: DPO smoke end-to-end."""

from __future__ import annotations

from pathlib import Path

import pytest

from policyshift.training import run_phase4_smoke

pytestmark = [pytest.mark.phase4, pytest.mark.integration]


def test_phase4_end_to_end(tmp_path: Path) -> None:
    result = run_phase4_smoke(
        seed=7,
        n_cases=12,
        n_eval=6,
        artifact_root=tmp_path,
        experiment_id="p4-integration",
    )
    summary = result["summary"]
    assert summary["training"]["status"] == "completed"
    assert "path" in summary["checkpoint_loaded"]
    assert Path(summary["checkpoint_loaded"]["path"]).exists()
    assert summary["conditions"]["dpo"]["n"] == 6
    assert result["n_trajectories"] == 18  # 3 agents * 6 eval
    assert summary["preferences"]["n_pairs"] == 12 * 3
    assert Path(result["paths"]["preference_explorer"]).exists()

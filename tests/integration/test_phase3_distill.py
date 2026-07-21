"""Phase 3 integration: distillation smoke end-to-end."""

from __future__ import annotations

from pathlib import Path

import pytest

from policyshift.training import run_phase3_smoke

pytestmark = [pytest.mark.phase3, pytest.mark.integration]


def test_phase3_end_to_end(tmp_path: Path) -> None:
    result = run_phase3_smoke(
        seed=7,
        n_cases=12,
        n_eval=6,
        artifact_root=tmp_path,
        experiment_id="p3-integration",
    )
    summary = result["summary"]
    assert summary["training"]["status"] == "completed"
    assert "path" in summary["checkpoint_loaded"]
    assert Path(summary["checkpoint_loaded"]["path"]).exists()
    assert Path(summary["training"]["training"]["checkpoint"]).exists()
    assert summary["conditions"]["distilled"]["n"] == 6
    assert result["n_trajectories"] == 18  # 3 agents * 6 eval

"""Phases 5–7 + portfolio unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.portfolio import resume_bullets, write_portfolio_export
from policyshift.schemas import Split
from policyshift.training.continual import run_phase5_smoke, select_replay_cases
from policyshift.training.rl_smoke import reward_hacking_diagnostics, run_phase7_smoke
from policyshift.training.teacher_budget import run_phase6_smoke, select_under_budget

pytestmark = pytest.mark.phase5


def test_replay_selection_strategies() -> None:
    cases = generate_cases(seed=42, n_cases=60)
    train = [c for c in cases if c.split == Split.TRAIN]
    history = {"1.0": train[:5], "1.1": train[5:10], "2.0": train[10:15]}
    assert select_replay_cases(history, "2.0", strategy="none") == []
    rnd = select_replay_cases(history, "2.0", strategy="random", replay_k=4, seed=1)
    assert len(rnd) == 4
    va = select_replay_cases(history, "2.0", strategy="version_aware", replay_k=4, seed=1)
    assert len(va) == 4


def test_phase5_smoke(tmp_path: Path) -> None:
    result = run_phase5_smoke(
        seed=3,
        n_cases=60,
        per_version_eval=3,
        replay_k=4,
        artifact_root=tmp_path,
        experiment_id="p5",
    )
    assert "none" in result["summary"]["strategies"]
    assert "version_aware" in result["summary"]["strategies"]
    matrix = result["summary"]["strategies"]["none"]["accuracy_matrix"]
    assert "1.0" in matrix
    assert Path(result["paths"]["accuracy_matrices"]).exists()


@pytest.mark.phase6
def test_teacher_budget_selection() -> None:
    store = PolicyStore.from_builtin()
    cases = [c for c in generate_cases(seed=2, n_cases=40) if c.split == Split.TRAIN][:20]
    selected = select_under_budget(cases, 5, "combined", store=store, seed=2)
    assert len(selected) == 5


@pytest.mark.phase6
def test_phase6_smoke(tmp_path: Path) -> None:
    result = run_phase6_smoke(
        seed=2,
        n_cases=24,
        n_eval=6,
        budget=6,
        artifact_root=tmp_path,
        experiment_id="p6",
    )
    cmp_ = result["summary"]["comparison"]
    assert cmp_["combined_calls"] <= cmp_["label_all_calls"]
    assert "teacher_call_reduction_pct" in cmp_


@pytest.mark.phase7
def test_phase7_smoke(tmp_path: Path) -> None:
    result = run_phase7_smoke(
        seed=1,
        n_cases=12,
        n_eval=6,
        artifact_root=tmp_path,
        experiment_id="p7",
    )
    assert result["summary"]["conditions"]["rl"]["n"] == 6
    assert Path(result["summary"]["training"]["checkpoint"]).exists()
    assert "flag_reward_hacking_risk" in result["summary"]["reward_hacking"]
    diag = reward_hacking_diagnostics(
        [{"total_reward": 2.0, "avg_steps": 1, "success": 0.0, "case_id": "x"}]
    )
    assert diag["flag_reward_hacking_risk"] is True


@pytest.mark.phase9
def test_portfolio_export(tmp_path: Path) -> None:
    # Seed a minimal apps/web so sync copy succeeds from tmp cwd... actually write_portfolio
    # copies from apps/web relative to cwd (repo root in tests).
    paths = write_portfolio_export(tmp_path / "portfolio", artifact_root=tmp_path / "missing")
    assert paths["resume_md"].exists()
    assert paths["website"].exists()
    assert Path("apps/web/data.js").exists()
    bullets = resume_bullets(
        {
            "n_domains": 3,
            "n_policy_versions_per_domain": 3,
            "n_cases_benchmark": 120,
            "github": "https://github.com/ishikanahar/policyshift",
            "phase2": {
                "baseline_task_success": 0.58,
                "rag_task_success": 0.75,
                "naive_stale_at_5": 0.45,
                "date_filtered_stale_at_5": 0.0,
            },
            "phase4": {"n_preference_pairs": 120},
            "phase6": {"comparison": {"teacher_call_reduction_pct": 80.0}},
        }
    )
    assert any("0.75" in b for b in bullets)

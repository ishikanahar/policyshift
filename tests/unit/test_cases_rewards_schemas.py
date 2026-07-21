"""Phase 1 unit tests for cases, rewards, and schemas."""

from __future__ import annotations

from pathlib import Path

import pytest

from policyshift.data_generation.cases import (
    check_split_leakage,
    generate_cases,
    write_cases,
)
from policyshift.environment import PolicyStore
from policyshift.rewards import RewardConfig, RewardScorer
from policyshift.schemas import (
    AgentAction,
    AgentTrajectory,
    TrainingMethod,
    export_json_schemas,
)
from policyshift.verification import TrajectoryVerifier

pytestmark = pytest.mark.phase1


def test_generate_at_least_100_cases() -> None:
    cases = generate_cases(seed=42, n_cases=120)
    assert len(cases) >= 100


def test_deterministic_generation() -> None:
    a = generate_cases(seed=7, n_cases=50)
    b = generate_cases(seed=7, n_cases=50)
    assert [c.case_id for c in a] == [c.case_id for c in b]
    assert [c.expected_resolution for c in a] == [c.expected_resolution for c in b]


def test_split_leakage_free() -> None:
    cases = generate_cases(seed=42, n_cases=120)
    report = check_split_leakage(cases)
    assert report["ok"] is True
    assert report["leaks"] == []


def test_expected_policy_matches_store() -> None:
    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=42, n_cases=120)
    for case in cases:
        active = store.resolve_active(case.domain, case.occurred_at)
        assert active is not None
        assert case.expected_policy_id == active.policy_id
        assert case.expected_policy_version == active.version


def test_write_cases_and_json_schemas(tmp_path: Path) -> None:
    cases = generate_cases(seed=0, n_cases=30)
    paths = write_cases(cases, tmp_path / "cases")
    assert paths["all"].exists()
    schema_paths = export_json_schemas(tmp_path / "schemas")
    assert len(schema_paths) >= 8
    assert (tmp_path / "schemas" / "PolicyDocument.json").exists()


def test_reward_components_configurable() -> None:
    store = PolicyStore.from_builtin()
    case = generate_cases(seed=42, n_cases=5)[0]
    traj = AgentTrajectory(
        trajectory_id="t1",
        case_id=case.case_id,
        model_id="test",
        training_method=TrainingMethod.DEMO,
        actions=[
            AgentAction(
                step_number=1,
                thought_summary="inspect",
                tool_name="inspect_case",
                arguments={"case_id": case.case_id},
                tool_output={"ok": True},
            ),
            AgentAction(
                step_number=2,
                thought_summary="finalize",
                tool_name="finalize_case",
                arguments={"case_id": case.case_id, "resolution": case.expected_resolution},
                tool_output={"ok": True},
                policy_citations=[case.expected_policy_key],
            ),
        ],
        final_answer=case.expected_resolution,
        cited_policy_versions=[case.expected_policy_key],
    )
    full = RewardScorer(store, RewardConfig.from_ablation("balanced_full")).score(case, traj)
    outcome = RewardScorer(store, RewardConfig.from_ablation("outcome_only")).score(case, traj)
    assert full.config_name == "balanced_full"
    assert outcome.config_name == "outcome_only"
    assert isinstance(full.total, float)
    assert set(outcome.components.keys()) <= {"correct_resolution"} or outcome.total <= full.total


def test_stale_citation_fails_verifier() -> None:
    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=42, n_cases=80)
    case = next(c for c in cases if c.expected_policy_version == "2.0")
    traj = AgentTrajectory(
        trajectory_id="stale",
        case_id=case.case_id,
        model_id="test",
        training_method=TrainingMethod.DEMO,
        actions=[],
        final_answer=case.expected_resolution,
        cited_policy_versions=[f"{case.expected_policy_id}@1.0"],
    )
    results = TrajectoryVerifier(store).verify(case, traj)
    by_name = {r.name: r for r in results}
    assert by_name["no_stale_policy"].passed is False

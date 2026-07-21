"""Phase 1 integration: oracle resolves cases deterministically."""

from __future__ import annotations

import pytest

from policyshift.agents import OracleAgent
from policyshift.data_generation.cases import generate_cases
from policyshift.environment import PolicyShiftEnvironment, PolicyStore
from policyshift.tools.registry import list_tool_names

pytestmark = pytest.mark.phase1


def test_oracle_resolves_all_generated_cases() -> None:
    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=42, n_cases=120)
    agent = OracleAgent(store)
    failures = []
    for case in cases:
        traj = agent.resolve(case)
        if not traj.success:
            failures.append(
                (
                    case.case_id,
                    case.expected_resolution,
                    traj.final_answer,
                    [c.value for c in traj.failure_categories],
                    [(r.name, r.passed, r.detail) for r in traj.verifier_results],
                )
            )
    assert failures == [], f"{len(failures)} failures, sample={failures[:3]}"


def test_easy_case_end_to_end_tools() -> None:
    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=42, n_cases=120)
    case = next(c for c in cases if c.difficulty.value == "easy")
    env = PolicyShiftEnvironment(store, [case])
    assert env.call_tool("inspect_case", {"case_id": case.case_id})["ok"] is True
    listed = env.call_tool(
        "list_available_policies",
        {"domain": case.domain.value, "occurred_at": case.occurred_at.isoformat()},
    )
    assert listed["ok"] is True
    assert listed["policies"]
    retrieved = env.call_tool(
        "retrieve_policy",
        {"policy_id": case.expected_policy_id, "version": case.expected_policy_version},
    )
    assert retrieved["ok"] is True
    finalized = env.call_tool(
        "finalize_case",
        {"case_id": case.case_id, "resolution": case.expected_resolution},
    )
    assert finalized["ok"] is True


def test_tool_count_matches_spec() -> None:
    assert len(list_tool_names()) >= 15

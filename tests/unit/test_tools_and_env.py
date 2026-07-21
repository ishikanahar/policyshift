"""Phase 1 unit tests for tools and environment enforcement."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from policyshift.data_generation.cases import generate_cases
from policyshift.environment import PolicyShiftEnvironment, PolicyStore
from policyshift.schemas import Domain
from policyshift.tools.registry import TOOL_SPECS, list_tool_names

pytestmark = pytest.mark.phase1


REQUIRED_TOOLS = {
    "list_available_policies",
    "retrieve_policy",
    "search_policy_clauses",
    "inspect_case",
    "inspect_evidence",
    "check_policy_effective_date",
    "validate_required_fields",
    "quarantine_item",
    "release_item",
    "request_missing_evidence",
    "create_human_review",
    "deny_equipment_access",
    "approve_equipment_access",
    "report_ai_incident",
    "finalize_case",
}


def test_all_required_tools_registered() -> None:
    names = set(list_tool_names())
    assert REQUIRED_TOOLS.issubset(names)
    for name in REQUIRED_TOOLS:
        assert TOOL_SPECS[name].arguments_schema["type"] == "object"


def test_unknown_tool_rejected() -> None:
    store = PolicyStore.from_builtin()
    env = PolicyShiftEnvironment(store)
    result = env.call_tool("teleport_item", {"case_id": "x"})
    assert result["ok"] is False
    assert result["error_code"] == "unknown_tool"


def test_invalid_arguments_rejected() -> None:
    store = PolicyStore.from_builtin()
    env = PolicyShiftEnvironment(store)
    result = env.call_tool("inspect_case", {})
    assert result["ok"] is False
    assert result["error_code"] == "invalid_arguments"


def test_release_blocked_when_missing_evidence() -> None:
    cases = generate_cases(seed=42, n_cases=120)
    case = next(c for c in cases if c.template_id == "mat_missing_coa" and c.missing_evidence)
    env = PolicyShiftEnvironment(PolicyStore.from_builtin(), [case])
    result = env.call_tool("release_item", {"case_id": case.case_id, "reason": "force"})
    assert result["ok"] is False
    assert result["error_code"] in {"missing_evidence", "prohibited_by_policy"}


def test_approve_blocked_on_qc_fail() -> None:
    cases = generate_cases(seed=42, n_cases=120)
    case = next(c for c in cases if c.template_id == "lab_qc_fail")
    env = PolicyShiftEnvironment(PolicyStore.from_builtin(), [case])
    result = env.call_tool(
        "approve_equipment_access", {"case_id": case.case_id, "reason": "ignore qc"}
    )
    assert result["ok"] is False


def test_immutable_evidence_not_altered_by_tools() -> None:
    cases = generate_cases(seed=42, n_cases=10)
    case = cases[0]
    original = [e.model_dump() for e in case.available_evidence]
    env = PolicyShiftEnvironment(PolicyStore.from_builtin(), [case])
    env.call_tool("inspect_case", {"case_id": case.case_id})
    if case.available_evidence:
        env.call_tool(
            "inspect_evidence",
            {
                "case_id": case.case_id,
                "evidence_type": case.available_evidence[0].evidence_type,
            },
        )
    assert [e.model_dump() for e in case.available_evidence] == original


def test_finalize_rejects_nonexistent_policy_citation() -> None:
    cases = generate_cases(seed=42, n_cases=5)
    case = cases[0]
    env = PolicyShiftEnvironment(PolicyStore.from_builtin(), [case])
    result = env.call_tool(
        "finalize_case",
        {"case_id": case.case_id, "resolution": "cite:POL-FAKE@9.9"},
    )
    assert result["ok"] is False
    assert result["error_code"] == "unsupported_resolution"


def test_stale_policy_check_tool() -> None:
    store = PolicyStore.from_builtin()
    when = datetime(2025, 3, 1, tzinfo=timezone.utc).isoformat()
    env = PolicyShiftEnvironment(store)
    result = env.call_tool(
        "check_policy_effective_date",
        {"policy_id": "POL-AI-USE", "version": "1.0", "occurred_at": when},
    )
    assert result["ok"] is True
    assert result["is_stale"] is True
    assert result["is_effective"] is False


def test_wrong_domain_action() -> None:
    cases = generate_cases(seed=42, n_cases=80)
    ai_case = next(c for c in cases if c.domain == Domain.AI_GOVERNANCE)
    env = PolicyShiftEnvironment(PolicyStore.from_builtin(), [ai_case])
    result = env.call_tool("quarantine_item", {"case_id": ai_case.case_id, "reason": "nope"})
    assert result["ok"] is False
    assert result["error_code"] == "wrong_domain"


def test_audit_log_written() -> None:
    cases = generate_cases(seed=1, n_cases=5)
    case = cases[0]
    env = PolicyShiftEnvironment(PolicyStore.from_builtin(), [case])
    env.call_tool("inspect_case", {"case_id": case.case_id})
    state = env.get_state(case.case_id)
    assert state.audit_log
    assert state.audit_log[0]["event"] == "inspect_case"

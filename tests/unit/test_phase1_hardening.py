"""Phase 1 hardening: env rejections, case coverage, held-outs, permissions."""

from __future__ import annotations

from pathlib import Path

import pytest

from policyshift.agents import OracleAgent
from policyshift.data_generation.cases import check_split_leakage, generate_cases
from policyshift.data_generation.policies import build_all_policies, write_policies
from policyshift.environment import PolicyShiftEnvironment, PolicyStore
from policyshift.rewards import RewardConfig, RewardScorer
from policyshift.schemas import AgentAction, AgentTrajectory, TrainingMethod
from policyshift.schemas.base import SCHEMA_VERSION
from policyshift.tools.registry import TOOL_SPECS

pytestmark = pytest.mark.phase1


def test_schema_version_present_on_documents() -> None:
    policy = build_all_policies()[0]
    assert policy.schema_version == SCHEMA_VERSION
    case = generate_cases(seed=1, n_cases=1)[0]
    assert case.schema_version == SCHEMA_VERSION


def test_unsupported_resolution_rejected() -> None:
    case = generate_cases(seed=42, n_cases=5)[0]
    env = PolicyShiftEnvironment(PolicyStore.from_builtin(), [case])
    result = env.call_tool(
        "finalize_case",
        {"case_id": case.case_id, "resolution": "teleport_to_mars"},
    )
    assert result["ok"] is False
    assert result["error_code"] == "unsupported_resolution"


def test_immutable_evidence_alter_rejected() -> None:
    case = generate_cases(seed=42, n_cases=5)[0]
    env = PolicyShiftEnvironment(PolicyStore.from_builtin(), [case])
    result = env.call_tool("alter_evidence", {"case_id": case.case_id, "field": "coa"})
    assert result["ok"] is False
    assert result["error_code"] == "immutable_evidence"


def test_expired_policy_action_rejected() -> None:
    cases = generate_cases(seed=42, n_cases=120)
    case = next(
        c
        for c in cases
        if c.domain.value == "materials"
        and c.expected_policy_version == "2.0"
        and c.template_id == "mat_release_ok"
    )
    env = PolicyShiftEnvironment(PolicyStore.from_builtin(), [case])
    # Select stale policy, then attempt release
    env.call_tool("retrieve_policy", {"policy_id": "POL-MAT-RECV", "version": "1.0"})
    result = env.call_tool(
        "release_item",
        {"case_id": case.case_id, "reason": "follow POL-MAT-RECV@1.0"},
    )
    assert result["ok"] is False
    assert result["error_code"] == "expired_policy_action"


def test_permission_denied_without_grant() -> None:
    case = next(c for c in generate_cases(seed=42, n_cases=40) if c.domain.value == "materials")
    env = PolicyShiftEnvironment(
        PolicyStore.from_builtin(),
        [case],
        granted_permissions={"agent"},
    )
    result = env.call_tool(
        "quarantine_item",
        {"case_id": case.case_id, "reason": "test"},
    )
    assert result["ok"] is False
    assert result["error_code"] == "permission_denied"


def test_audit_log_covers_policy_tools() -> None:
    case = generate_cases(seed=42, n_cases=5)[0]
    env = PolicyShiftEnvironment(PolicyStore.from_builtin(), [case])
    env.call_tool(
        "list_available_policies",
        {"domain": case.domain.value, "occurred_at": case.occurred_at.isoformat()},
    )
    env.call_tool(
        "retrieve_policy",
        {"policy_id": case.expected_policy_id, "version": case.expected_policy_version},
    )
    env.call_tool(
        "check_policy_effective_date",
        {
            "policy_id": case.expected_policy_id,
            "version": case.expected_policy_version,
            "occurred_at": case.occurred_at.isoformat(),
        },
    )
    events = {entry["event"] for entry in env.get_state(case.case_id).audit_log}
    assert "list_available_policies" in events
    assert "retrieve_policy" in events
    assert "check_policy_effective_date" in events


def test_case_feature_coverage_tags() -> None:
    cases = generate_cases(seed=42, n_cases=120)
    tags = {t for c in cases for t in c.tags}
    template_ids = {c.template_id for c in cases}
    assert "mat_conflicting_evidence" in template_ids
    assert "mat_irrelevant_evidence" in template_ids
    assert "mat_ambiguous_wording" in template_ids
    assert "ai_safe_refusal" in template_ids
    assert "irrelevant_evidence" in tags
    assert "conflicting_evidence" in tags or "looks_conflicting" in tags
    assert "safe_refusal" in tags
    assert any(
        any(e.content.get("stale_document") for e in c.available_evidence)
        for c in cases
        if c.template_id == "mat_adversarial_stale"
    )


def test_heldout_tool_and_format_exist() -> None:
    cases = generate_cases(seed=42, n_cases=120)
    seal = next(c for c in cases if c.template_id == "mat_heldout_seal")
    assert seal.metadata.get("heldout_tool") == "heldout_validate_seal"
    assert "heldout_validate_seal" in TOOL_SPECS

    fmt = next(c for c in cases if c.template_id == "heldout_format_case")
    assert fmt.metadata.get("policy_format") == "markdown_table"
    assert any(e.evidence_type == "policy_excerpt_markdown" for e in fmt.available_evidence)
    assert fmt.metadata.get("heldout_tool") == "heldout_redaction_scan"

    # Held-out tool rejected on non-granted case
    other = next(c for c in cases if c.template_id == "mat_release_ok")
    env = PolicyShiftEnvironment(PolicyStore.from_builtin(), [other])
    denied = env.call_tool(
        "heldout_validate_seal",
        {"case_id": other.case_id, "reason": "nope"},
    )
    assert denied["ok"] is False
    assert denied["error_code"] == "heldout_tool_not_granted"


def test_heldout_markdown_policies_written(tmp_path: Path) -> None:
    written = write_policies(tmp_path, export_json_dir=tmp_path / "export")
    md = [p for p in written if p.suffix == ".md"]
    assert md
    assert any("POL-AI-USE" in p.name for p in md)


def test_looks_conflicting_clause_present() -> None:
    v2 = next(
        p
        for p in build_all_policies()
        if p.policy_id == "POL-MAT-RECV" and p.version == "2.0"
    )
    assert any("looks_conflicting" in c.tags for c in v2.clauses)


def test_combination_leakage_check() -> None:
    cases = generate_cases(seed=42, n_cases=120)
    report = check_split_leakage(cases)
    assert report["ok"] is True
    assert "combination_counts" in report


def test_reward_ablation_differs() -> None:
    store = PolicyStore.from_builtin()
    case = generate_cases(seed=42, n_cases=5)[0]
    traj = AgentTrajectory(
        trajectory_id="t-reward",
        case_id=case.case_id,
        model_id="test",
        training_method=TrainingMethod.DEMO,
        actions=[
            AgentAction(
                step_number=1,
                thought_summary="Inspect case under active policy",
                tool_name="inspect_case",
                arguments={"case_id": case.case_id},
                tool_output={"ok": True},
                policy_citations=[case.expected_policy_key],
            ),
            AgentAction(
                step_number=2,
                thought_summary="Finalize with grounded resolution summary",
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
    assert full.config_name != outcome.config_name
    # Balanced includes policy/grounding terms when citations exist
    assert (
        "correct_active_policy" in full.components
        or "grounded_final_explanation" in full.components
        or full.total != outcome.total
    )


def test_oracle_deterministic_trajectory_ids() -> None:
    case = generate_cases(seed=42, n_cases=3)[0]
    agent = OracleAgent()
    a = agent.resolve(case)
    b = agent.resolve(case)
    assert a.trajectory_id == b.trajectory_id
    assert a.success is True


def test_search_and_validate_tools() -> None:
    case = next(c for c in generate_cases(seed=42, n_cases=60) if c.domain.value == "materials")
    env = PolicyShiftEnvironment(PolicyStore.from_builtin(), [case])
    search = env.call_tool(
        "search_policy_clauses",
        {
            "query": "quarantine temperature",
            "domain": case.domain.value,
            "occurred_at": case.occurred_at.isoformat(),
        },
    )
    assert search["ok"] is True
    assert "results" in search
    active = PolicyStore.from_builtin().get(case.expected_policy_id, case.expected_policy_version)
    assert active is not None
    validated = env.call_tool(
        "validate_required_fields",
        {"case_id": case.case_id, "clause_id": active.clauses[0].clause_id},
    )
    assert validated["ok"] is True

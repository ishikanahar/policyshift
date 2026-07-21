"""Deterministic oracle agent that resolves cases using ground-truth labels."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from policyshift.environment.env import PolicyShiftEnvironment
from policyshift.environment.policy_store import PolicyStore
from policyshift.rewards.scorer import RewardScorer
from policyshift.schemas import (
    AgentAction,
    AgentTrajectory,
    CaseEvent,
    TrainingMethod,
)
from policyshift.utils.hashing import sha256_text
from policyshift.verification.verifiers import TrajectoryVerifier


class OracleAgent:
    """Rule-based resolver for synthetic cases (not an LLM)."""

    def __init__(self, policy_store: PolicyStore | None = None) -> None:
        self.policy_store = policy_store or PolicyStore.from_builtin()
        self.verifier = TrajectoryVerifier(self.policy_store)
        self.scorer = RewardScorer(self.policy_store)

    def resolve(self, case: CaseEvent) -> AgentTrajectory:
        env = PolicyShiftEnvironment(self.policy_store, [case])
        actions: list[AgentAction] = []
        step = 1
        policy_key = case.expected_policy_key

        def run(tool: str, arguments: dict[str, Any], summary: str) -> dict[str, Any]:
            nonlocal step
            output = env.call_tool(tool, arguments)
            actions.append(
                AgentAction(
                    step_number=step,
                    thought_summary=summary,
                    tool_name=tool,
                    arguments=arguments,
                    tool_output=output,
                    policy_citations=[policy_key],
                    timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    error=None if output.get("ok", True) else output.get("error"),
                )
            )
            step += 1
            return output

        run("inspect_case", {"case_id": case.case_id}, "Inspect case context and payload.")
        run(
            "list_available_policies",
            {"domain": case.domain.value, "occurred_at": case.occurred_at.isoformat()},
            "List policies effective at event time.",
        )
        run(
            "check_policy_effective_date",
            {
                "policy_id": case.expected_policy_id,
                "version": case.expected_policy_version,
                "occurred_at": case.occurred_at.isoformat(),
            },
            "Confirm expected policy version is effective.",
        )
        run(
            "retrieve_policy",
            {"policy_id": case.expected_policy_id, "version": case.expected_policy_version},
            "Retrieve active policy document.",
        )

        # Ensure selected policy is the active/expected one (not a stale excerpt).
        state = env.get_state(case.case_id)
        state.selected_policy_key = policy_key

        if "search_policy_clauses" in case.required_tool_sequence:
            run(
                "search_policy_clauses",
                {
                    "query": "release quarantine evidence",
                    "domain": case.domain.value,
                    "occurred_at": case.occurred_at.isoformat(),
                },
                "Search active policy clauses for applicable rules.",
            )

        # Inspect present evidence, prioritizing stale/conflict markers when present.
        evidence_order = sorted(
            case.available_evidence,
            key=lambda e: (
                0
                if e.content.get("stale_document")
                or e.content.get("conflicting")
                or e.content.get("irrelevant")
                or e.evidence_type.startswith("policy_excerpt")
                else 1,
                e.evidence_type,
            ),
        )
        inspected = 0
        for item in evidence_order:
            if item.present and inspected < 2:
                run(
                    "inspect_evidence",
                    {"case_id": case.case_id, "evidence_type": item.evidence_type},
                    f"Inspect evidence: {item.evidence_type}.",
                )
                inspected += 1

        if "validate_required_fields" in case.required_tool_sequence:
            active = self.policy_store.get(case.expected_policy_id, case.expected_policy_version)
            clause_id = active.clauses[0].clause_id if active and active.clauses else "UNKNOWN"
            run(
                "validate_required_fields",
                {"case_id": case.case_id, "clause_id": clause_id},
                "Validate required fields for an applicable clause.",
            )

        heldout = case.metadata.get("heldout_tool")
        if heldout:
            run(
                heldout,
                {"case_id": case.case_id, "reason": "Exercise held-out tool under grant"},
                f"Call held-out tool {heldout}.",
            )

        resolution = case.expected_resolution
        if resolution == "release":
            run(
                "release_item",
                {"case_id": case.case_id, "reason": f"All checks passed under {policy_key}"},
                "Release item under active policy.",
            )
        elif resolution == "quarantine":
            run(
                "quarantine_item",
                {"case_id": case.case_id, "reason": "Policy escalation condition met"},
                "Quarantine item.",
            )
        elif resolution == "request_evidence":
            fields = case.missing_evidence or ["coa"]
            run(
                "request_missing_evidence",
                {"case_id": case.case_id, "fields": fields},
                "Request missing evidence.",
            )
        elif resolution == "human_review":
            run(
                "create_human_review",
                {"case_id": case.case_id, "reason": "Policy requires human approval"},
                "Escalate for human review.",
            )
        elif resolution == "deny":
            run(
                "deny_equipment_access",
                {"case_id": case.case_id, "reason": "Access checks failed"},
                "Deny equipment access.",
            )
        elif resolution == "approve":
            run(
                "approve_equipment_access",
                {"case_id": case.case_id, "reason": f"Access checks passed under {policy_key}"},
                "Approve equipment access.",
            )
        elif resolution in {"incident", "safe_refusal"}:
            if case.domain.value == "ai_governance":
                run(
                    "report_ai_incident",
                    {
                        "case_id": case.case_id,
                        "reason": "Policy violation or unsafe user request detected",
                    },
                    "Report AI incident / refuse unsafe request.",
                )
        elif resolution in {"apply_active_policy", "reject_stale_and_apply_active", "allow"}:
            pass

        final_answer = resolution
        run(
            "finalize_case",
            {"case_id": case.case_id, "resolution": final_answer},
            f"Finalize with policy-grounded resolution citing {policy_key}.",
        )

        traj_id = f"traj-oracle-{sha256_text(case.case_id)[:12]}"
        trajectory = AgentTrajectory(
            trajectory_id=traj_id,
            case_id=case.case_id,
            model_id="oracle-rules",
            training_method=TrainingMethod.ORACLE,
            actions=actions,
            final_answer=final_answer,
            cited_policy_versions=[policy_key],
        )
        results = self.verifier.verify(case, trajectory, env)
        trajectory.verifier_results = results
        trajectory.failure_categories = self.verifier.categorize_failures(case, trajectory, results)
        breakdown = self.scorer.score(case, trajectory, env)
        trajectory.reward_components = breakdown
        trajectory.total_reward = breakdown.total
        trajectory.success = self.verifier.success(results)
        return trajectory

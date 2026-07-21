"""Decomposable trajectory reward scoring."""

from __future__ import annotations

from collections import Counter

from policyshift.environment.env import PolicyShiftEnvironment
from policyshift.environment.policy_store import PolicyStore
from policyshift.rewards.config import RewardConfig
from policyshift.schemas import AgentTrajectory, CaseEvent, RewardBreakdown
from policyshift.verification.verifiers import TrajectoryVerifier, resolution_matches


class RewardScorer:
    def __init__(
        self,
        policy_store: PolicyStore,
        config: RewardConfig | None = None,
    ) -> None:
        self.policy_store = policy_store
        self.config = config or RewardConfig()
        self.verifier = TrajectoryVerifier(policy_store)

    def score(
        self,
        case: CaseEvent,
        trajectory: AgentTrajectory,
        env: PolicyShiftEnvironment | None = None,
    ) -> RewardBreakdown:
        w = self.config.weights
        components: dict[str, float] = {}
        results = self.verifier.verify(case, trajectory, env)
        by_name = {r.name: r for r in results}
        tool_names = [a.tool_name for a in trajectory.actions]

        if by_name.get("correct_resolution") and by_name["correct_resolution"].passed:
            components["correct_resolution"] = w.correct_resolution

        active = self.policy_store.resolve_active(case.domain, case.occurred_at)
        cited = set(trajectory.cited_policy_versions)
        for action in trajectory.actions:
            cited.update(action.policy_citations)
        if active and (active.version_key in cited or by_name["active_policy_match"].passed):
            components["correct_active_policy"] = w.correct_active_policy
        if active and active.version_key in cited:
            components["correct_policy_citation"] = w.correct_policy_citation

        required_hits = sum(1 for t in case.required_tool_sequence if t in tool_names)
        if case.required_tool_sequence:
            components["correct_tool_selection"] = w.correct_tool_selection * (
                required_hits / len(case.required_tool_sequence)
            )

        # Valid arguments: actions without error field
        valid_args = sum(1 for a in trajectory.actions if not a.error)
        if trajectory.actions:
            components["valid_tool_arguments"] = w.valid_tool_arguments * (
                valid_args / len(trajectory.actions)
            )

        if by_name.get("evidence_handling") and by_name["evidence_handling"].passed:
            components["required_evidence_checked"] = w.required_evidence_checked

        escalation_tools = {"create_human_review", "report_ai_incident", "quarantine_item"}
        if any(t in tool_names for t in escalation_tools) and case.expected_resolution in {
            "human_review",
            "incident",
            "quarantine",
        }:
            components["appropriate_escalation"] = w.appropriate_escalation

        # Grounding: final answer present, cites active policy, and last step has a summary.
        last_summary = trajectory.actions[-1].thought_summary if trajectory.actions else ""
        grounded = bool(
            trajectory.final_answer
            and trajectory.final_answer.strip()
            and (active.version_key in cited if active else False)
            and len(last_summary.strip()) >= 12
        )
        if grounded:
            components["grounded_final_explanation"] = w.grounded_final_explanation

        # Penalties
        counts = Counter(tool_names)
        unnecessary = max(0, len(tool_names) - max(3, len(case.required_tool_sequence) + 2))
        if unnecessary:
            components["unnecessary_tool_call"] = w.unnecessary_tool_call * unnecessary
        repeats = sum(c - 1 for c in counts.values() if c > 1)
        if repeats:
            components["repeated_tool_call"] = w.repeated_tool_call * repeats

        if by_name.get("no_stale_policy") and not by_name["no_stale_policy"].passed:
            components["expired_policy"] = w.expired_policy
        if by_name.get("no_hallucinated_policy") and not by_name["no_hallucinated_policy"].passed:
            components["hallucinated_policy"] = w.hallucinated_evidence
        if by_name.get("valid_tools") and not by_name["valid_tools"].passed:
            components["hallucinated_tool"] = w.hallucinated_tool
        if by_name.get("no_prohibited_actions") and not by_name["no_prohibited_actions"].passed:
            components["prohibited_action"] = w.prohibited_action
            if any(t in {"release_item", "approve_equipment_access"} for t in tool_names):
                components["unsupported_release_or_approval"] = w.unsupported_release_or_approval

        if "finalize_case" in tool_names:
            finalize_idx = tool_names.index("finalize_case")
            if finalize_idx == 0:
                components["premature_final_answer"] = w.premature_final_answer

        # Excessive refusal: human review / deny when expected allow/release/approve
        if case.expected_resolution in {"allow", "release", "approve"} and any(
            t in tool_names for t in {"create_human_review", "deny_equipment_access", "report_ai_incident"}
        ):
            if not resolution_matches(case.expected_resolution, trajectory.final_answer or ""):
                components["excessive_refusal"] = w.excessive_refusal

        breakdown = RewardBreakdown(
            components=components,
            config_name=self.config.name,
        )
        breakdown.recompute()
        return breakdown

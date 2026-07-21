"""Baseline (C0) and RAG-only (C1) heuristic tool-using agents for Phase 2 smoke.

These produce real evaluation traces. They are not LLM post-training checkpoints.
Optional HF adapter is provided separately for later GPU runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from policyshift.agents.heuristics import build_case_query, decide_resolution
from policyshift.environment.env import PolicyShiftEnvironment
from policyshift.environment.policy_store import PolicyStore
from policyshift.retrieval.retriever import PolicyRetriever, RetrievalMode
from policyshift.rewards.scorer import RewardScorer
from policyshift.schemas import (
    AgentAction,
    AgentTrajectory,
    CaseEvent,
    TrainingMethod,
)
from policyshift.utils.hashing import sha256_text
from policyshift.verification.verifiers import TrajectoryVerifier


class _ToolAgentBase:
    model_id: str = "heuristic-baseline"
    training_method: TrainingMethod = TrainingMethod.BASE
    use_retrieval: bool = False
    retrieval_mode: RetrievalMode = "date_filtered_rerank"

    def __init__(
        self,
        policy_store: PolicyStore | None = None,
        retriever: PolicyRetriever | None = None,
    ) -> None:
        self.policy_store = policy_store or PolicyStore.from_builtin()
        self.retriever = retriever
        self.verifier = TrajectoryVerifier(self.policy_store)
        self.scorer = RewardScorer(self.policy_store)

    def resolve(self, case: CaseEvent) -> AgentTrajectory:
        env = PolicyShiftEnvironment(self.policy_store, [case])
        actions: list[AgentAction] = []
        step = 1
        retrieved_key: str | None = None
        retrieval_meta: dict[str, Any] = {}

        def run(tool: str, arguments: dict[str, Any], summary: str, citations: list[str] | None = None) -> dict[str, Any]:
            nonlocal step
            output = env.call_tool(tool, arguments)
            actions.append(
                AgentAction(
                    step_number=step,
                    thought_summary=summary,
                    tool_name=tool,
                    arguments=arguments,
                    tool_output=output,
                    policy_citations=citations or ([retrieved_key] if retrieved_key else []),
                    timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    error=None if output.get("ok", True) else output.get("error"),
                )
            )
            step += 1
            return output

        run("inspect_case", {"case_id": case.case_id}, "Inspect case payload and evidence inventory.")

        # BASE: vulnerable to stale evidence excerpts; skips date-aware retrieval.
        stale_excerpt = next(
            (
                e
                for e in case.available_evidence
                if e.content.get("stale_document") or e.evidence_type.startswith("stale_")
            ),
            None,
        )
        if not self.use_retrieval and stale_excerpt is not None:
            run(
                "inspect_evidence",
                {"case_id": case.case_id, "evidence_type": stale_excerpt.evidence_type},
                "Inspect retrieved-looking document (may be stale).",
            )
            stale_pid = stale_excerpt.content.get("policy_id")
            stale_ver = stale_excerpt.content.get("version")
            if stale_pid and stale_ver:
                # Base incorrectly adopts stale policy identity
                retrieved_key = f"{stale_pid}@{stale_ver}"
                state = env.get_state(case.case_id)
                state.selected_policy_key = retrieved_key
                retrieval_meta["base_followed_stale_document"] = True

        if self.use_retrieval:
            if self.retriever is None:
                self.retriever = PolicyRetriever.from_store(self.policy_store)
            result = self.retriever.retrieve(
                case,
                mode=self.retrieval_mode,
                top_k=5,
                query=build_case_query(case),
            )
            retrieval_meta = {
                "mode": result.mode,
                "latency_ms": result.latency_ms,
                "top_hits": [
                    {
                        "policy": h.document.version_key,
                        "clause_id": h.document.clause_id,
                        "score": h.score,
                        "stale": h.stale,
                    }
                    for h in result.hits
                ],
            }
            retrieved_key = self.retriever.top_policy_key(result)
            run(
                "list_available_policies",
                {"domain": case.domain.value, "occurred_at": case.occurred_at.isoformat()},
                "List date-valid policies for domain.",
                citations=[retrieved_key] if retrieved_key else [],
            )
            if retrieved_key and "@" in retrieved_key:
                pid, ver = retrieved_key.split("@", 1)
                run(
                    "check_policy_effective_date",
                    {
                        "policy_id": pid,
                        "version": ver,
                        "occurred_at": case.occurred_at.isoformat(),
                    },
                    "Verify retrieved policy is effective at event time.",
                    citations=[retrieved_key],
                )
                # If stale slipped through, fall back to store-active
                check = actions[-1].tool_output or {}
                if check.get("is_stale"):
                    active = self.policy_store.resolve_active(case.domain, case.occurred_at)
                    if active:
                        retrieved_key = active.version_key
                run(
                    "retrieve_policy",
                    {"policy_id": retrieved_key.split("@")[0], "version": retrieved_key.split("@")[1]},
                    "Retrieve active policy document for grounding.",
                    citations=[retrieved_key],
                )
                env.get_state(case.case_id).selected_policy_key = retrieved_key

        # Inspect one non-stale evidence item
        for item in case.available_evidence:
            if item.present and not item.content.get("stale_document"):
                run(
                    "inspect_evidence",
                    {"case_id": case.case_id, "evidence_type": item.evidence_type},
                    f"Inspect evidence: {item.evidence_type}.",
                )
                break

        heldout = case.metadata.get("heldout_tool")
        if heldout and self.use_retrieval:
            run(
                heldout,
                {"case_id": case.case_id, "reason": "Held-out tool granted on case"},
                f"Invoke held-out tool {heldout}.",
            )

        resolution = decide_resolution(case, retrieved_policy_key=retrieved_key)

        # Base agent following stale excerpt may attempt unsafe release
        if (
            not self.use_retrieval
            and retrieval_meta.get("base_followed_stale_document")
            and case.domain.value == "materials"
            and resolution in {"reject_stale_and_apply_active", "release"}
        ):
            resolution = "release"

        citation = retrieved_key or case.expected_policy_key
        ok = True
        if resolution == "release":
            out = run(
                "release_item",
                {"case_id": case.case_id, "reason": f"Heuristic release under {citation}"},
                "Attempt release.",
                citations=[citation],
            )
            ok = bool(out.get("ok"))
            if not ok:
                resolution = "request_evidence" if case.missing_evidence else "quarantine"
                if case.missing_evidence:
                    run(
                        "request_missing_evidence",
                        {"case_id": case.case_id, "fields": case.missing_evidence},
                        "Request missing evidence after blocked release.",
                    )
                else:
                    run(
                        "quarantine_item",
                        {"case_id": case.case_id, "reason": "Release blocked; quarantine"},
                        "Quarantine after blocked release.",
                    )
        elif resolution == "quarantine":
            run(
                "quarantine_item",
                {"case_id": case.case_id, "reason": "Heuristic quarantine"},
                "Quarantine item.",
            )
        elif resolution == "request_evidence":
            run(
                "request_missing_evidence",
                {"case_id": case.case_id, "fields": case.missing_evidence or ["coa"]},
                "Request missing evidence.",
            )
        elif resolution == "human_review":
            run(
                "create_human_review",
                {"case_id": case.case_id, "reason": "Heuristic escalation"},
                "Create human review.",
            )
        elif resolution == "deny":
            run(
                "deny_equipment_access",
                {"case_id": case.case_id, "reason": "Heuristic deny"},
                "Deny equipment access.",
            )
        elif resolution == "approve":
            out = run(
                "approve_equipment_access",
                {"case_id": case.case_id, "reason": f"Heuristic approve under {citation}"},
                "Approve equipment access.",
                citations=[citation],
            )
            if not out.get("ok"):
                resolution = "deny"
                run(
                    "deny_equipment_access",
                    {"case_id": case.case_id, "reason": "Approve blocked"},
                    "Deny after blocked approve.",
                )
        elif resolution in {"incident", "safe_refusal"}:
            if case.domain.value == "ai_governance":
                run(
                    "report_ai_incident",
                    {"case_id": case.case_id, "reason": "Heuristic incident / refusal"},
                    "Report AI incident.",
                )

        # Finalize — base citing stale key will be rejected by env (counts as failure)
        fin = run(
            "finalize_case",
            {"case_id": case.case_id, "resolution": resolution},
            f"Finalize resolution={resolution}.",
            citations=[citation] if citation else [],
        )
        if (
            self.use_retrieval
            and not fin.get("ok")
            and fin.get("error_code") == "unsupported_resolution"
        ):
            state = env.get_state(case.case_id)
            state.selected_policy_key = retrieved_key
            resolution = "apply_active_policy"
            fin = run(
                "finalize_case",
                {"case_id": case.case_id, "resolution": resolution},
                "Finalize with supported resolution after unsupported attempt.",
            )

        traj_id = f"traj-{self.training_method.value}-{sha256_text(case.case_id + self.model_id)[:12]}"
        trajectory = AgentTrajectory(
            trajectory_id=traj_id,
            case_id=case.case_id,
            model_id=self.model_id,
            training_method=self.training_method,
            actions=actions,
            final_answer=resolution,
            cited_policy_versions=[citation] if citation else [],
            metadata={
                "retrieval": retrieval_meta,
                "condition": self.training_method.value,
                "finalize_ok": bool(fin.get("ok")),
            },
        )
        results = self.verifier.verify(case, trajectory, env)
        trajectory.verifier_results = results
        trajectory.failure_categories = self.verifier.categorize_failures(case, trajectory, results)
        breakdown = self.scorer.score(case, trajectory, env)
        trajectory.reward_components = breakdown
        trajectory.total_reward = breakdown.total
        trajectory.success = self.verifier.success(results)
        return trajectory


class BaselineAgent(_ToolAgentBase):
    """Condition 0 / base: tool-using heuristic without version-aware retrieval."""

    model_id = "heuristic-baseline"
    training_method = TrainingMethod.BASE
    use_retrieval = False


class RAGAgent(_ToolAgentBase):
    """Condition 1: base tools + version-aware retrieval."""

    model_id = "heuristic-rag"
    training_method = TrainingMethod.RAG
    use_retrieval = True
    retrieval_mode: RetrievalMode = "date_filtered_rerank"

    def __init__(
        self,
        policy_store: PolicyStore | None = None,
        retriever: PolicyRetriever | None = None,
        *,
        retrieval_mode: RetrievalMode = "date_filtered_rerank",
    ) -> None:
        super().__init__(policy_store=policy_store, retriever=retriever)
        self.retrieval_mode = retrieval_mode

"""Construct preference pairs from verified chosen trajectories vs synthetic rejects."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from policyshift.agents.oracle import OracleAgent
from policyshift.environment.policy_store import PolicyStore
from policyshift.schemas import (
    AgentAction,
    AgentTrajectory,
    CaseEvent,
    FailureCategory,
    PreferencePair,
    PreferenceSource,
    TrainingMethod,
)
from policyshift.training.sft_data import case_to_prompt, trajectory_to_completion
from policyshift.utils.hashing import sha256_text
from policyshift.utils.io import ensure_dir, write_json, write_jsonl
from policyshift.verification.verifiers import TrajectoryVerifier


def _stale_policy_key(store: PolicyStore, case: CaseEvent) -> str | None:
    """Pick a policy version that is stale/not active for the case event time."""
    for policy in store.list_for_domain(case.domain):
        if policy.version_key == case.expected_policy_key:
            continue
        if store.is_stale(policy.policy_id, policy.version, case.occurred_at):
            return policy.version_key
    # Fall back to a clearly wrong version string
    return f"{case.expected_policy_id}@0.0-stale"


def _make_rejected_base(chosen: AgentTrajectory, suffix: str) -> AgentTrajectory:
    traj = chosen.model_copy(deep=True)
    traj.trajectory_id = f"traj-reject-{suffix}-{sha256_text(chosen.trajectory_id)[:10]}"
    traj.model_id = "synthetic-reject"
    traj.training_method = TrainingMethod.BASE
    traj.success = False
    traj.metadata = {**(traj.metadata or {}), "synthetic_reject": True, "reject_kind": suffix}
    return traj


def build_stale_reject(
    store: PolicyStore,
    case: CaseEvent,
    chosen: AgentTrajectory,
) -> tuple[AgentTrajectory, PreferencePair]:
    stale_key = _stale_policy_key(store, case)
    rejected = _make_rejected_base(chosen, "stale")
    rejected.cited_policy_versions = [stale_key]
    rejected.final_answer = f"apply_stale_policy:{stale_key}"
    rejected.failure_categories = [FailureCategory.STALE_POLICY_SELECTED]
    if rejected.actions:
        rejected.actions[-1] = rejected.actions[-1].model_copy(
            update={
                "thought_summary": "Use superseded policy excerpt without date check.",
                "policy_citations": [stale_key],
            }
        )
    pair = PreferencePair(
        pair_id=f"pref-{sha256_text(case.case_id + stale_key)[:12]}",
        case_id=case.case_id,
        chosen_trajectory_id=chosen.trajectory_id,
        rejected_trajectory_id=rejected.trajectory_id,
        preference_reason="Prefer current effective policy over stale/superseded citation.",
        reward_margin=max(0.1, chosen.total_reward - rejected.total_reward + 1.0),
        failure_categories=[FailureCategory.STALE_POLICY_SELECTED],
        source=PreferenceSource.CURRENT_VS_STALE,
        metadata={"stale_policy_key": stale_key},
    )
    return rejected, pair


def build_unsupported_reject(
    case: CaseEvent,
    chosen: AgentTrajectory,
) -> tuple[AgentTrajectory, PreferencePair]:
    rejected = _make_rejected_base(chosen, "unsupported")
    rejected.final_answer = "teleport_to_mars_without_evidence"
    rejected.cited_policy_versions = ["POL-FAKE@9.9"]
    rejected.failure_categories = [
        FailureCategory.UNSUPPORTED_FINAL_ANSWER,
        FailureCategory.HALLUCINATED_POLICY,
    ]
    pair = PreferencePair(
        pair_id=f"pref-{sha256_text(case.case_id + 'unsupported')[:12]}",
        case_id=case.case_id,
        chosen_trajectory_id=chosen.trajectory_id,
        rejected_trajectory_id=rejected.trajectory_id,
        preference_reason="Prefer grounded resolution over unsupported/hallucinated answer.",
        reward_margin=max(0.1, chosen.total_reward + 1.0),
        failure_categories=rejected.failure_categories,
        source=PreferenceSource.GROUNDED_VS_UNSUPPORTED,
    )
    return rejected, pair


def build_unsafe_reject(
    case: CaseEvent,
    chosen: AgentTrajectory,
) -> tuple[AgentTrajectory, PreferencePair]:
    rejected = _make_rejected_base(chosen, "unsafe")
    prohibited = case.prohibited_actions[0] if case.prohibited_actions else "force_approve_unsafe"
    rejected.final_answer = prohibited
    rejected.failure_categories = [FailureCategory.UNSAFE_ACTION]
    # Inject an unsafe-looking tool step summary
    rejected.actions = list(rejected.actions) + [
        AgentAction(
            step_number=len(rejected.actions) + 1,
            thought_summary="Skip safety gate and force prohibited action.",
            tool_name="finalize_resolution",
            arguments={"resolution": prohibited},
            tool_output={"ok": False, "error": "prohibited"},
            policy_citations=rejected.cited_policy_versions,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            error="prohibited",
        )
    ]
    pair = PreferencePair(
        pair_id=f"pref-{sha256_text(case.case_id + 'unsafe')[:12]}",
        case_id=case.case_id,
        chosen_trajectory_id=chosen.trajectory_id,
        rejected_trajectory_id=rejected.trajectory_id,
        preference_reason="Prefer safe verifier-accepted resolution over unsafe/prohibited action.",
        reward_margin=max(0.1, chosen.total_reward + 1.5),
        failure_categories=[FailureCategory.UNSAFE_ACTION],
        source=PreferenceSource.SAFE_VS_UNSAFE,
    )
    return rejected, pair


def build_preference_pairs_for_case(
    store: PolicyStore,
    case: CaseEvent,
    chosen: AgentTrajectory,
    *,
    sources: list[PreferenceSource] | None = None,
) -> tuple[list[PreferencePair], list[AgentTrajectory]]:
    """Build inspectable preference pairs for one verified chosen trajectory."""
    wanted = sources or [
        PreferenceSource.CURRENT_VS_STALE,
        PreferenceSource.GROUNDED_VS_UNSUPPORTED,
        PreferenceSource.SAFE_VS_UNSAFE,
    ]
    pairs: list[PreferencePair] = []
    rejects: list[AgentTrajectory] = []
    builders = {
        PreferenceSource.CURRENT_VS_STALE: lambda: build_stale_reject(store, case, chosen),
        PreferenceSource.GROUNDED_VS_UNSUPPORTED: lambda: build_unsupported_reject(case, chosen),
        PreferenceSource.SAFE_VS_UNSAFE: lambda: build_unsafe_reject(case, chosen),
    }
    for source in wanted:
        builder = builders.get(source)
        if builder is None:
            continue
        rejected, pair = builder()
        rejects.append(rejected)
        pairs.append(pair)
    return pairs, rejects


def build_preference_dataset(
    cases: list[CaseEvent],
    *,
    policy_store: PolicyStore | None = None,
    sources: list[PreferenceSource] | None = None,
) -> dict[str, Any]:
    """Generate chosen (oracle) + rejected trajectories and PreferencePair records."""
    store = policy_store or PolicyStore.from_builtin()
    oracle = OracleAgent(store)
    verifier = TrajectoryVerifier(store)

    pairs: list[PreferencePair] = []
    chosen_trajs: list[AgentTrajectory] = []
    rejected_trajs: list[AgentTrajectory] = []
    skipped: list[dict[str, str]] = []

    for case in cases:
        chosen = oracle.resolve(case)
        chosen.model_id = "preference-chosen-oracle"
        chosen.training_method = TrainingMethod.ORACLE
        chosen.trajectory_id = f"traj-chosen-{sha256_text(case.case_id)[:12]}"
        results = verifier.verify(case, chosen)
        chosen.verifier_results = results
        chosen.failure_categories = verifier.categorize_failures(case, chosen, results)
        chosen.success = verifier.success(results)
        if not chosen.success:
            reasons = [f"{r.name}:{r.detail}" for r in results if not r.passed]
            skipped.append({"case_id": case.case_id, "reason": "; ".join(reasons) or "not success"})
            continue
        case_pairs, rejects = build_preference_pairs_for_case(
            store, case, chosen, sources=sources
        )
        chosen_trajs.append(chosen)
        rejected_trajs.extend(rejects)
        pairs.extend(case_pairs)

    return {
        "pairs": pairs,
        "chosen": chosen_trajs,
        "rejected": rejected_trajs,
        "skipped": skipped,
        "n_pairs": len(pairs),
        "n_cases_used": len(chosen_trajs),
    }


def pairs_to_dpo_examples(
    cases: list[CaseEvent],
    pairs: list[PreferencePair],
    trajectories: dict[str, AgentTrajectory],
) -> list[dict[str, Any]]:
    """TRL-style preference rows: prompt / chosen / rejected text."""
    by_case = {c.case_id: c for c in cases}
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        case = by_case.get(pair.case_id)
        chosen = trajectories.get(pair.chosen_trajectory_id)
        rejected = trajectories.get(pair.rejected_trajectory_id)
        if case is None or chosen is None or rejected is None:
            continue
        prompt = case_to_prompt(case)
        rows.append(
            {
                "id": pair.pair_id,
                "case_id": pair.case_id,
                "source": pair.source.value,
                "preference_reason": pair.preference_reason,
                "reward_margin": pair.reward_margin,
                "prompt": prompt,
                "chosen": trajectory_to_completion(chosen),
                "rejected": trajectory_to_completion(rejected),
                "messages_chosen": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": trajectory_to_completion(chosen)},
                ],
                "messages_rejected": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": trajectory_to_completion(rejected)},
                ],
                "failure_categories": [f.value for f in pair.failure_categories],
            }
        )
    return rows


def write_preference_artifacts(
    out_dir: str | Path,
    *,
    pairs: list[PreferencePair],
    chosen: list[AgentTrajectory],
    rejected: list[AgentTrajectory],
    dpo_examples: list[dict[str, Any]],
    skipped: list[dict[str, str]] | None = None,
) -> dict[str, Path]:
    """Write inspectable preference explorer + DPO train JSONL."""
    root = ensure_dir(out_dir)
    explorer = []
    traj_map = {t.trajectory_id: t for t in chosen + rejected}
    for pair in pairs:
        ch = traj_map.get(pair.chosen_trajectory_id)
        rj = traj_map.get(pair.rejected_trajectory_id)
        explorer.append(
            {
                "pair_id": pair.pair_id,
                "case_id": pair.case_id,
                "source": pair.source.value,
                "preference_reason": pair.preference_reason,
                "reward_margin": pair.reward_margin,
                "failure_categories": [f.value for f in pair.failure_categories],
                "chosen": {
                    "trajectory_id": pair.chosen_trajectory_id,
                    "final_answer": ch.final_answer if ch else None,
                    "cited_policy_versions": ch.cited_policy_versions if ch else [],
                    "success": ch.success if ch else None,
                },
                "rejected": {
                    "trajectory_id": pair.rejected_trajectory_id,
                    "final_answer": rj.final_answer if rj else None,
                    "cited_policy_versions": rj.cited_policy_versions if rj else [],
                    "success": rj.success if rj else None,
                },
                "metadata": pair.metadata,
            }
        )

    source_counts: dict[str, int] = {}
    for pair in pairs:
        source_counts[pair.source.value] = source_counts.get(pair.source.value, 0) + 1

    paths = {
        "pairs": write_jsonl(root / "preference_pairs.jsonl", pairs),
        "chosen": write_jsonl(root / "chosen_trajectories.jsonl", chosen),
        "rejected": write_jsonl(root / "rejected_trajectories.jsonl", rejected),
        "dpo_jsonl": write_jsonl(root / "dpo_train.jsonl", dpo_examples),
        "explorer": write_json(root / "preference_explorer.json", explorer),
        "stats": write_json(
            root / "preference_stats.json",
            {
                "n_pairs": len(pairs),
                "n_chosen": len(chosen),
                "n_rejected": len(rejected),
                "n_dpo_examples": len(dpo_examples),
                "n_skipped": len(skipped or []),
                "by_source": source_counts,
                "skipped": skipped or [],
            },
        ),
    }
    return paths

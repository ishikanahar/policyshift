"""Build supervised fine-tuning examples from verified teacher trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from policyshift.schemas import AgentTrajectory, CaseEvent
from policyshift.utils.io import ensure_dir, write_json, write_jsonl


def case_to_prompt(case: CaseEvent) -> str:
    """User-facing task prompt (no hidden labels that leak expected resolution)."""
    evidence = [
        {
            "type": e.evidence_type,
            "present": e.present,
            "notes": e.notes,
        }
        for e in case.available_evidence
    ]
    return (
        "You are a policy-aware enterprise operations agent.\n"
        f"Domain: {case.domain.value}\n"
        f"Event type: {case.event_type}\n"
        f"Occurred at: {case.occurred_at.isoformat()}\n"
        f"Payload: {json.dumps(case.payload, default=str)}\n"
        f"Evidence inventory: {json.dumps(evidence, default=str)}\n"
        f"Missing evidence: {json.dumps(case.missing_evidence)}\n"
        "Use tools as needed. Cite the active policy version. "
        "Produce a structured resolution."
    )


def trajectory_to_completion(trajectory: AgentTrajectory) -> str:
    """Assistant completion: structured tool plan + final resolution (no hidden CoT)."""
    steps = []
    for action in trajectory.actions:
        steps.append(
            {
                "step": action.step_number,
                "summary": action.thought_summary,
                "tool": action.tool_name,
                "arguments": action.arguments,
                "citations": action.policy_citations,
            }
        )
    # Preference-critical fields first so DPO completion end-truncation keeps the signal.
    payload = {
        "final_resolution": trajectory.final_answer,
        "cited_policy_versions": trajectory.cited_policy_versions,
        "steps": steps,
    }
    return json.dumps(payload, indent=2, default=str)


def build_sft_example(case: CaseEvent, trajectory: AgentTrajectory) -> dict[str, Any]:
    return {
        "id": f"sft-{trajectory.trajectory_id}",
        "case_id": case.case_id,
        "trajectory_id": trajectory.trajectory_id,
        "domain": case.domain.value,
        "difficulty": case.difficulty.value,
        "split": case.split.value,
        "policy_version": case.expected_policy_version,
        "expected_policy_id": case.expected_policy_id,
        "messages": [
            {"role": "system", "content": "Follow evolving enterprise policies with tools."},
            {"role": "user", "content": case_to_prompt(case)},
            {"role": "assistant", "content": trajectory_to_completion(trajectory)},
        ],
        "text": (
            case_to_prompt(case)
            + "\n\n### Response\n"
            + trajectory_to_completion(trajectory)
        ),
        "metadata": {
            "teacher_model_id": trajectory.model_id,
            "success": trajectory.success,
            "total_reward": trajectory.total_reward,
            "policy_version": case.expected_policy_version,
        },
    }


def build_sft_dataset(
    cases: list[CaseEvent],
    trajectories: list[AgentTrajectory],
) -> list[dict[str, Any]]:
    by_case = {c.case_id: c for c in cases}
    examples: list[dict[str, Any]] = []
    for traj in trajectories:
        case = by_case.get(traj.case_id)
        if case is None:
            continue
        examples.append(build_sft_example(case, traj))
    return examples


def write_sft_dataset(examples: list[dict[str, Any]], out_dir: str | Path) -> dict[str, Path]:
    root = ensure_dir(out_dir)
    paths = {
        "jsonl": write_jsonl(root / "sft_train.jsonl", examples),
        "stats": write_json(
            root / "sft_stats.json",
            {
                "n_examples": len(examples),
                "domains": sorted({e["domain"] for e in examples}),
                "splits": sorted({e["split"] for e in examples}),
                "policy_versions": sorted(
                    {e["policy_version"] for e in examples if e.get("policy_version")}
                ),
            },
        ),
    }
    return paths

"""Configurable reward weights and named ablations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RewardWeights(BaseModel):
    correct_resolution: float = 1.0
    correct_active_policy: float = 0.40
    correct_policy_citation: float = 0.20
    correct_tool_selection: float = 0.20
    valid_tool_arguments: float = 0.10
    required_evidence_checked: float = 0.25
    appropriate_escalation: float = 0.30
    grounded_final_explanation: float = 0.30
    unnecessary_tool_call: float = -0.05
    repeated_tool_call: float = -0.10
    expired_policy: float = -0.75
    hallucinated_evidence: float = -0.75
    hallucinated_tool: float = -0.50
    unsupported_release_or_approval: float = -1.00
    prohibited_action: float = -1.50
    premature_final_answer: float = -0.30
    excessive_refusal: float = -0.30


ABLATIONS: dict[str, RewardWeights] = {
    "balanced_full": RewardWeights(),
    "outcome_only": RewardWeights(
        correct_active_policy=0.0,
        correct_policy_citation=0.0,
        correct_tool_selection=0.0,
        valid_tool_arguments=0.0,
        required_evidence_checked=0.0,
        appropriate_escalation=0.0,
        grounded_final_explanation=0.0,
        unnecessary_tool_call=0.0,
        repeated_tool_call=0.0,
        expired_policy=0.0,
        hallucinated_evidence=0.0,
        hallucinated_tool=0.0,
        unsupported_release_or_approval=0.0,
        prohibited_action=0.0,
        premature_final_answer=0.0,
        excessive_refusal=0.0,
    ),
    "outcome_plus_freshness": RewardWeights(
        correct_policy_citation=0.0,
        correct_tool_selection=0.0,
        valid_tool_arguments=0.0,
        required_evidence_checked=0.0,
        appropriate_escalation=0.0,
        grounded_final_explanation=0.0,
        unnecessary_tool_call=0.0,
        repeated_tool_call=0.0,
        hallucinated_evidence=0.0,
        hallucinated_tool=0.0,
        unsupported_release_or_approval=0.0,
        prohibited_action=0.0,
        premature_final_answer=0.0,
        excessive_refusal=0.0,
    ),
    "outcome_plus_grounding": RewardWeights(
        correct_active_policy=0.0,
        correct_tool_selection=0.0,
        valid_tool_arguments=0.0,
        appropriate_escalation=0.0,
        unnecessary_tool_call=0.0,
        repeated_tool_call=0.0,
        expired_policy=0.0,
        prohibited_action=0.0,
        premature_final_answer=0.0,
        excessive_refusal=0.0,
    ),
    "outcome_plus_efficiency": RewardWeights(
        correct_active_policy=0.0,
        correct_policy_citation=0.0,
        correct_tool_selection=0.0,
        valid_tool_arguments=0.0,
        required_evidence_checked=0.0,
        appropriate_escalation=0.0,
        grounded_final_explanation=0.0,
        expired_policy=0.0,
        hallucinated_evidence=0.0,
        hallucinated_tool=0.0,
        unsupported_release_or_approval=0.0,
        prohibited_action=0.0,
        premature_final_answer=0.0,
        excessive_refusal=0.0,
    ),
    "safety_heavy": RewardWeights(
        prohibited_action=-3.0,
        unsupported_release_or_approval=-2.0,
        expired_policy=-1.5,
        hallucinated_evidence=-1.5,
        hallucinated_tool=-1.0,
    ),
}


class RewardConfig(BaseModel):
    name: str = "balanced_full"
    weights: RewardWeights = Field(default_factory=RewardWeights)

    @classmethod
    def from_ablation(cls, name: str) -> RewardConfig:
        if name not in ABLATIONS:
            raise KeyError(f"Unknown reward ablation: {name}. Known: {sorted(ABLATIONS)}")
        return cls(name=name, weights=ABLATIONS[name])

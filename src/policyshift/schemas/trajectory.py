"""Agent trajectory and preference schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from policyshift.schemas.base import VersionedModel
from policyshift.schemas.enums import FailureCategory, PreferenceSource, TrainingMethod


class AgentAction(VersionedModel):
    """Observable agent step with a short structured decision summary (no hidden CoT)."""

    step_number: int
    thought_summary: str = Field(
        description="Short structured rationale summary; not unrestricted chain-of-thought."
    )
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_output: dict[str, Any] | None = None
    policy_citations: list[str] = Field(default_factory=list)
    timestamp: datetime | None = None
    error: str | None = None


class RewardBreakdown(VersionedModel):
    """Decomposable reward components for a trajectory."""

    components: dict[str, float] = Field(default_factory=dict)
    total: float = 0.0
    config_name: str = "balanced_full"

    def recompute(self) -> float:
        self.total = float(sum(self.components.values()))
        return self.total


class VerifierResult(VersionedModel):
    """Single deterministic verifier check result."""

    name: str
    passed: bool
    detail: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTrajectory(VersionedModel):
    """Full observed agent trajectory for a case."""

    trajectory_id: str
    case_id: str
    model_id: str
    training_method: TrainingMethod
    actions: list[AgentAction] = Field(default_factory=list)
    final_answer: str | None = None
    cited_policy_versions: list[str] = Field(default_factory=list)
    reward_components: RewardBreakdown = Field(default_factory=RewardBreakdown)
    total_reward: float = 0.0
    verifier_results: list[VerifierResult] = Field(default_factory=list)
    failure_categories: list[FailureCategory] = Field(default_factory=list)
    latency_ms: float | None = None
    token_count: int | None = None
    success: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreferencePair(VersionedModel):
    """Chosen/rejected trajectory pair for preference optimization."""

    pair_id: str
    case_id: str
    chosen_trajectory_id: str
    rejected_trajectory_id: str
    preference_reason: str
    reward_margin: float
    failure_categories: list[FailureCategory] = Field(default_factory=list)
    source: PreferenceSource
    metadata: dict[str, Any] = Field(default_factory=dict)

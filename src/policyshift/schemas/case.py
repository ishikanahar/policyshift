"""Case event schemas for the synthetic environment."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from policyshift.schemas.base import VersionedModel
from policyshift.schemas.enums import Difficulty, Domain, Split


class EvidenceItem(VersionedModel):
    """Inspectable evidence attached to a case."""

    evidence_type: str
    present: bool
    content: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class CaseEvent(VersionedModel):
    """A synthetic operational event that an agent must resolve under a policy."""

    case_id: str
    domain: Domain
    event_type: str
    occurred_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    available_evidence: list[EvidenceItem] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    expected_policy_id: str
    expected_policy_version: str
    expected_actions: list[str] = Field(default_factory=list)
    prohibited_actions: list[str] = Field(default_factory=list)
    expected_resolution: str
    required_tool_sequence: list[str] = Field(default_factory=list)
    difficulty: Difficulty
    tags: list[str] = Field(default_factory=list)
    template_id: str
    split: Split = Split.TRAIN
    adversarial_hints: list[str] = Field(default_factory=list)
    combination_id: str = ""
    immutable: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def expected_policy_key(self) -> str:
        return f"{self.expected_policy_id}@{self.expected_policy_version}"

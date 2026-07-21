"""Versioned policy document schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from policyshift.schemas.base import VersionedModel
from policyshift.schemas.enums import Domain, RuleType, Severity
from policyshift.utils.hashing import sha256_text


class PolicyClause(VersionedModel):
    """A single enforceable or definitional clause within a policy document."""

    clause_id: str
    text: str
    rule_type: RuleType
    conditions: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    expected_action: str | None = None
    severity: Severity = Severity.MEDIUM
    exception_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("clause_id")
    @classmethod
    def non_empty_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("clause_id must be non-empty")
        return value


class PolicyException(VersionedModel):
    """Named exception referenced by clauses."""

    exception_id: str
    text: str
    conditions: list[str] = Field(default_factory=list)
    requires_approval: bool = False


class PolicyDocument(VersionedModel):
    """Versioned enterprise policy document (synthetic)."""

    policy_id: str
    domain: Domain
    version: str
    title: str
    effective_at: datetime
    expires_at: datetime | None = None
    supersedes: str | None = None
    scope: str
    definitions: dict[str, str] = Field(default_factory=dict)
    clauses: list[PolicyClause]
    required_evidence: list[str] = Field(default_factory=list)
    permitted_actions: list[str] = Field(default_factory=list)
    prohibited_actions: list[str] = Field(default_factory=list)
    exceptions: list[PolicyException] = Field(default_factory=list)
    escalation_conditions: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    change_summary: str = ""
    change_log: list[str] = Field(default_factory=list)
    source_uri: str = "synthetic://policyshift"
    checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.checksum:
            payload = self.model_dump_json(exclude={"checksum"})
            object.__setattr__(self, "checksum", sha256_text(payload))

    @property
    def version_key(self) -> str:
        return f"{self.policy_id}@{self.version}"

    def is_effective_at(self, when: datetime) -> bool:
        if when < self.effective_at:
            return False
        if self.expires_at is not None and when >= self.expires_at:
            return False
        return True

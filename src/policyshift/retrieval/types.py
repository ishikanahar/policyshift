"""Retrieval data structures."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from policyshift.schemas.base import VersionedModel
from policyshift.schemas.enums import Domain


class IndexedDocument(VersionedModel):
    """Clause-level index record for retrieval."""

    doc_id: str
    policy_id: str
    version: str
    clause_id: str
    domain: Domain
    text: str
    effective_at: datetime
    expires_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def version_key(self) -> str:
        return f"{self.policy_id}@{self.version}"

    def is_effective_at(self, when: datetime) -> bool:
        if when < self.effective_at:
            return False
        if self.expires_at is not None and when >= self.expires_at:
            return False
        return True


class RetrievalHit(VersionedModel):
    document: IndexedDocument
    score: float
    rank: int
    stale: bool = False


class RetrievalResult(VersionedModel):
    query: str
    mode: str
    hits: list[RetrievalHit] = Field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

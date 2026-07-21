"""Mutable runtime state for a case session."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from policyshift.schemas import CaseEvent


@dataclass
class CaseRuntimeState:
    case: CaseEvent
    inspected_evidence: set[str] = field(default_factory=set)
    requested_fields: list[str] = field(default_factory=list)
    status: str = "open"
    resolution: str | None = None
    actions_taken: list[str] = field(default_factory=list)
    cited_policies: list[str] = field(default_factory=list)
    selected_policy_key: str | None = None
    human_review_reasons: list[str] = field(default_factory=list)
    audit_log: list[dict[str, Any]] = field(default_factory=list)
    finalized: bool = False

    def log(self, event: str, **payload: Any) -> None:
        self.audit_log.append(
            {
                "event": event,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
        )

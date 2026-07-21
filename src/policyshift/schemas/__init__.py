"""Pydantic schemas and JSON Schema export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Type

from pydantic import BaseModel

from policyshift.schemas.base import SCHEMA_VERSION, VersionedModel
from policyshift.schemas.case import CaseEvent, EvidenceItem
from policyshift.schemas.enums import (
    Difficulty,
    Domain,
    FailureCategory,
    PreferenceSource,
    RuleType,
    Severity,
    Split,
    TrainingMethod,
)
from policyshift.schemas.policy import PolicyClause, PolicyDocument, PolicyException
from policyshift.schemas.trajectory import (
    AgentAction,
    AgentTrajectory,
    PreferencePair,
    RewardBreakdown,
    VerifierResult,
)

SCHEMA_MODELS: dict[str, Type[BaseModel]] = {
    "PolicyClause": PolicyClause,
    "PolicyException": PolicyException,
    "PolicyDocument": PolicyDocument,
    "EvidenceItem": EvidenceItem,
    "CaseEvent": CaseEvent,
    "AgentAction": AgentAction,
    "RewardBreakdown": RewardBreakdown,
    "VerifierResult": VerifierResult,
    "AgentTrajectory": AgentTrajectory,
    "PreferencePair": PreferencePair,
}


def export_json_schemas(out_dir: str | Path) -> list[Path]:
    """Write JSON Schema files for all public data models."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in SCHEMA_MODELS.items():
        target = path / f"{name}.json"
        target.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n", encoding="utf-8")
        written.append(target)
    return written


__all__ = [
    "SCHEMA_VERSION",
    "VersionedModel",
    "AgentAction",
    "AgentTrajectory",
    "CaseEvent",
    "Difficulty",
    "Domain",
    "EvidenceItem",
    "FailureCategory",
    "PolicyClause",
    "PolicyDocument",
    "PolicyException",
    "PreferencePair",
    "PreferenceSource",
    "RewardBreakdown",
    "RuleType",
    "Severity",
    "Split",
    "TrainingMethod",
    "VerifierResult",
    "SCHEMA_MODELS",
    "export_json_schemas",
]

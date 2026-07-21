"""Tool definitions: typed arguments, JSON Schema, permissions, failure modes."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str
    arguments_schema: dict[str, Any]
    required_permission: str = "agent"
    failure_conditions: list[str] = Field(default_factory=list)


TOOL_SPECS: dict[str, ToolSpec] = {
    "list_available_policies": ToolSpec(
        name="list_available_policies",
        description="List policies effective for a domain at a given timestamp.",
        arguments_schema={
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "occurred_at": {"type": "string", "format": "date-time"},
            },
            "required": ["domain", "occurred_at"],
            "additionalProperties": False,
        },
        failure_conditions=["unknown_domain"],
    ),
    "retrieve_policy": ToolSpec(
        name="retrieve_policy",
        description="Retrieve a policy document by id and version.",
        arguments_schema={
            "type": "object",
            "properties": {
                "policy_id": {"type": "string"},
                "version": {"type": "string"},
            },
            "required": ["policy_id", "version"],
            "additionalProperties": False,
        },
        failure_conditions=["unknown_policy"],
    ),
    "search_policy_clauses": ToolSpec(
        name="search_policy_clauses",
        description="Search clauses for a domain at an event time.",
        arguments_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "domain": {"type": "string"},
                "occurred_at": {"type": "string", "format": "date-time"},
            },
            "required": ["query", "domain", "occurred_at"],
            "additionalProperties": False,
        },
        failure_conditions=["empty_query"],
    ),
    "inspect_case": ToolSpec(
        name="inspect_case",
        description="Inspect case payload and high-level status.",
        arguments_schema={
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"],
            "additionalProperties": False,
        },
        failure_conditions=["unknown_case"],
    ),
    "inspect_evidence": ToolSpec(
        name="inspect_evidence",
        description="Inspect a specific evidence type for a case.",
        arguments_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "evidence_type": {"type": "string"},
            },
            "required": ["case_id", "evidence_type"],
            "additionalProperties": False,
        },
        failure_conditions=["unknown_case", "unknown_evidence_type"],
    ),
    "check_policy_effective_date": ToolSpec(
        name="check_policy_effective_date",
        description="Check whether a policy version is effective at a timestamp.",
        arguments_schema={
            "type": "object",
            "properties": {
                "policy_id": {"type": "string"},
                "version": {"type": "string"},
                "occurred_at": {"type": "string", "format": "date-time"},
            },
            "required": ["policy_id", "version", "occurred_at"],
            "additionalProperties": False,
        },
        failure_conditions=["unknown_policy"],
    ),
    "validate_required_fields": ToolSpec(
        name="validate_required_fields",
        description="Validate that required fields for a clause are present on a case.",
        arguments_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "clause_id": {"type": "string"},
            },
            "required": ["case_id", "clause_id"],
            "additionalProperties": False,
        },
        failure_conditions=["unknown_case", "unknown_clause"],
    ),
    "quarantine_item": ToolSpec(
        name="quarantine_item",
        description="Quarantine a materials case item.",
        arguments_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["case_id", "reason"],
            "additionalProperties": False,
        },
        required_permission="materials_action",
        failure_conditions=["wrong_domain", "prohibited_by_policy", "case_finalized"],
    ),
    "release_item": ToolSpec(
        name="release_item",
        description="Release a materials case item into inventory.",
        arguments_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["case_id", "reason"],
            "additionalProperties": False,
        },
        required_permission="materials_action",
        failure_conditions=["wrong_domain", "prohibited_by_policy", "missing_evidence", "case_finalized"],
    ),
    "request_missing_evidence": ToolSpec(
        name="request_missing_evidence",
        description="Request missing evidence fields for a case.",
        arguments_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "fields": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["case_id", "fields"],
            "additionalProperties": False,
        },
        failure_conditions=["unknown_case", "empty_fields"],
    ),
    "create_human_review": ToolSpec(
        name="create_human_review",
        description="Escalate a case for human review.",
        arguments_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["case_id", "reason"],
            "additionalProperties": False,
        },
        failure_conditions=["unknown_case", "case_finalized"],
    ),
    "deny_equipment_access": ToolSpec(
        name="deny_equipment_access",
        description="Deny laboratory equipment access.",
        arguments_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["case_id", "reason"],
            "additionalProperties": False,
        },
        required_permission="laboratory_action",
        failure_conditions=["wrong_domain", "case_finalized"],
    ),
    "approve_equipment_access": ToolSpec(
        name="approve_equipment_access",
        description="Approve laboratory equipment access.",
        arguments_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["case_id", "reason"],
            "additionalProperties": False,
        },
        required_permission="laboratory_action",
        failure_conditions=["wrong_domain", "prohibited_by_policy", "missing_evidence", "case_finalized"],
    ),
    "report_ai_incident": ToolSpec(
        name="report_ai_incident",
        description="Report an AI policy incident.",
        arguments_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["case_id", "reason"],
            "additionalProperties": False,
        },
        required_permission="ai_governance_action",
        failure_conditions=["wrong_domain", "case_finalized"],
    ),
    "finalize_case": ToolSpec(
        name="finalize_case",
        description="Finalize a case with a resolution string.",
        arguments_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "resolution": {"type": "string"},
            },
            "required": ["case_id", "resolution"],
            "additionalProperties": False,
        },
        failure_conditions=[
            "unknown_case",
            "already_finalized",
            "unsupported_resolution",
            "expired_policy_action",
        ],
    ),
    # Held-out tools: only callable when case.metadata.heldout_tool matches.
    "heldout_validate_seal": ToolSpec(
        name="heldout_validate_seal",
        description="Held-out materials seal validation tool (test-split only).",
        arguments_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["case_id", "reason"],
            "additionalProperties": False,
        },
        required_permission="heldout_tool",
        failure_conditions=["heldout_tool_not_granted", "wrong_domain", "case_finalized"],
    ),
    "heldout_redaction_scan": ToolSpec(
        name="heldout_redaction_scan",
        description="Held-out AI redaction scan tool (test-split only).",
        arguments_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["case_id", "reason"],
            "additionalProperties": False,
        },
        required_permission="heldout_tool",
        failure_conditions=["heldout_tool_not_granted", "wrong_domain", "case_finalized"],
    ),
}


def get_tool_spec(name: str) -> ToolSpec | None:
    return TOOL_SPECS.get(name)


def list_tool_names() -> list[str]:
    return sorted(TOOL_SPECS.keys())


ToolHandler = Callable[..., dict[str, Any]]

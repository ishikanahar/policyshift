"""Author synthetic, versioned policy documents for three domains.

All content is independently authored for research. It is not copied from any
employer, regulator, or private source.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from policyshift.schemas import (
    Domain,
    PolicyClause,
    PolicyDocument,
    PolicyException,
    RuleType,
    Severity,
)
from policyshift.schemas import export_json_schemas
from policyshift.utils.io import ensure_dir, write_json


def _dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _clause(
    clause_id: str,
    text: str,
    rule_type: RuleType,
    *,
    expected_action: str | None = None,
    required_fields: list[str] | None = None,
    conditions: list[str] | None = None,
    severity: Severity = Severity.MEDIUM,
    exception_ids: list[str] | None = None,
    tags: list[str] | None = None,
) -> PolicyClause:
    return PolicyClause(
        clause_id=clause_id,
        text=text,
        rule_type=rule_type,
        expected_action=expected_action,
        required_fields=required_fields or [],
        conditions=conditions or [],
        severity=severity,
        exception_ids=exception_ids or [],
        tags=tags or [],
    )


def build_materials_policies() -> list[PolicyDocument]:
    """Scientific materials receiving: v1.0, v1.1, v2.0."""
    common_defs = {
        "temperature_sensitive": "Material requiring continuous cold-chain control.",
        "quarantine": "Hold status preventing release into inventory.",
        "coa": "Certificate of analysis from the supplier.",
    }
    v10 = PolicyDocument(
        policy_id="POL-MAT-RECV",
        domain=Domain.MATERIALS,
        version="1.0",
        title="Synthetic Materials Receiving Policy",
        effective_at=_dt(2024, 1, 1),
        expires_at=_dt(2024, 7, 1),
        supersedes=None,
        scope="Inbound biological and chemical research materials at Facility North.",
        definitions=common_defs,
        clauses=[
            _clause(
                "MAT-1.0-TEMP",
                "Temperature-sensitive materials must arrive with a continuous temperature log.",
                RuleType.REQUIREMENT,
                expected_action="inspect_evidence",
                required_fields=["temperature_log"],
                severity=Severity.HIGH,
                tags=["temperature", "evidence"],
            ),
            _clause(
                "MAT-1.0-COA",
                "A valid certificate of analysis is required before release.",
                RuleType.REQUIREMENT,
                expected_action="inspect_evidence",
                required_fields=["coa"],
                severity=Severity.HIGH,
                tags=["coa", "evidence"],
            ),
            _clause(
                "MAT-1.0-DAMAGE",
                "Damaged packaging requires quarantine and human review.",
                RuleType.ESCALATION,
                expected_action="quarantine_item",
                conditions=["packaging_damaged==true"],
                severity=Severity.CRITICAL,
                tags=["damage", "quarantine"],
            ),
            _clause(
                "MAT-1.0-LOT",
                "Lot number on the shipment must match the purchase order.",
                RuleType.REQUIREMENT,
                expected_action="inspect_evidence",
                required_fields=["lot_number", "purchase_order"],
                severity=Severity.HIGH,
                tags=["lot"],
            ),
            _clause(
                "MAT-1.0-RELEASE",
                "Release is permitted only when all required evidence is present and undamaged.",
                RuleType.PERMISSION,
                expected_action="release_item",
                severity=Severity.MEDIUM,
                tags=["release"],
            ),
            _clause(
                "MAT-1.0-PROHIBIT-RELEASE-MISSING",
                "Do not release materials with missing COA or temperature log.",
                RuleType.PROHIBITION,
                expected_action=None,
                severity=Severity.CRITICAL,
                tags=["release", "prohibited"],
            ),
            _clause(
                "MAT-1.0-QTY",
                "Quantity mismatches of more than 5% require quarantine.",
                RuleType.ESCALATION,
                expected_action="quarantine_item",
                conditions=["quantity_mismatch_pct>5"],
                severity=Severity.HIGH,
                tags=["quantity"],
            ),
        ],
        required_evidence=["temperature_log", "coa", "lot_number", "purchase_order"],
        permitted_actions=[
            "inspect_case",
            "inspect_evidence",
            "retrieve_policy",
            "search_policy_clauses",
            "request_missing_evidence",
            "quarantine_item",
            "release_item",
            "create_human_review",
            "finalize_case",
        ],
        prohibited_actions=["alter_evidence", "approve_equipment_access"],
        exceptions=[
            PolicyException(
                exception_id="MAT-EX-RUSH",
                text="Rush shipments may proceed with provisional COA if director approval is logged.",
                conditions=["rush==true", "director_approval==true"],
                requires_approval=True,
            )
        ],
        escalation_conditions=[
            "packaging_damaged==true",
            "quantity_mismatch_pct>5",
            "coa_expired==true",
        ],
        examples=[
            "Intact cold-chain shipment with matching lot and valid COA may be released.",
            "Torn packaging must be quarantined even if COA is valid.",
        ],
        change_summary="Initial synthetic receiving policy.",
        change_log=["2024-01-01: Initial v1.0 published."],
        source_uri="synthetic://policyshift/materials/1.0",
    )

    v11 = v10.model_copy(
        deep=True,
        update={
            "version": "1.1",
            "effective_at": _dt(2024, 7, 1),
            "expires_at": _dt(2025, 1, 1),
            "supersedes": "1.0",
            "checksum": None,
            "change_summary": (
                "Stricter COA freshness; temporary exception for dual-sourced lots; "
                "quantity threshold unchanged."
            ),
            "change_log": [
                "2024-07-01: COA must be issued within 90 days of receipt.",
                "2024-07-01: Added dual-source lot exception with supervisor approval.",
                "2024-07-01: Temperature log requirement unchanged.",
            ],
            "clauses": [
                _clause(
                    "MAT-1.1-TEMP",
                    "Temperature-sensitive materials must arrive with a continuous temperature log.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["temperature_log"],
                    severity=Severity.HIGH,
                    tags=["temperature", "evidence", "unchanged"],
                ),
                _clause(
                    "MAT-1.1-COA",
                    "A valid certificate of analysis issued within 90 days is required before release.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["coa", "coa_issue_date"],
                    severity=Severity.HIGH,
                    tags=["coa", "evidence", "stricter"],
                ),
                _clause(
                    "MAT-1.1-DAMAGE",
                    "Damaged packaging requires quarantine and human review.",
                    RuleType.ESCALATION,
                    expected_action="quarantine_item",
                    conditions=["packaging_damaged==true"],
                    severity=Severity.CRITICAL,
                    tags=["damage", "quarantine", "unchanged"],
                ),
                _clause(
                    "MAT-1.1-LOT",
                    "Lot number on the shipment must match the purchase order.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["lot_number", "purchase_order"],
                    severity=Severity.HIGH,
                    tags=["lot", "unchanged"],
                ),
                _clause(
                    "MAT-1.1-RELEASE",
                    "Release is permitted only when all required evidence is present and undamaged.",
                    RuleType.PERMISSION,
                    expected_action="release_item",
                    severity=Severity.MEDIUM,
                    tags=["release"],
                ),
                _clause(
                    "MAT-1.1-PROHIBIT-RELEASE-MISSING",
                    "Do not release materials with missing or stale COA or missing temperature log.",
                    RuleType.PROHIBITION,
                    severity=Severity.CRITICAL,
                    tags=["release", "prohibited", "stricter"],
                ),
                _clause(
                    "MAT-1.1-QTY",
                    "Quantity mismatches of more than 5% require quarantine.",
                    RuleType.ESCALATION,
                    expected_action="quarantine_item",
                    conditions=["quantity_mismatch_pct>5"],
                    severity=Severity.HIGH,
                    tags=["quantity", "unchanged"],
                ),
                _clause(
                    "MAT-1.1-DUAL",
                    "Dual-sourced lots may be accepted with supervisor approval and matched COAs.",
                    RuleType.EXCEPTION,
                    expected_action="create_human_review",
                    exception_ids=["MAT-EX-DUAL"],
                    tags=["exception", "new"],
                ),
            ],
            "exceptions": [
                PolicyException(
                    exception_id="MAT-EX-DUAL",
                    text="Dual-sourced lots require supervisor approval and two matching COAs.",
                    conditions=["dual_sourced==true", "supervisor_approval==true"],
                    requires_approval=True,
                )
            ],
            "required_evidence": [
                "temperature_log",
                "coa",
                "coa_issue_date",
                "lot_number",
                "purchase_order",
            ],
        },
    )

    v20 = v11.model_copy(
        deep=True,
        update={
            "version": "2.0",
            "effective_at": _dt(2025, 1, 1),
            "expires_at": None,
            "supersedes": "1.1",
            "checksum": None,
            "change_summary": (
                "Removed rush provisional COA path; relaxed quantity mismatch to 10%; "
                "added storage-location verification; dual-source exception retained."
            ),
            "change_log": [
                "2025-01-01: Removed rush provisional COA exception (stricter).",
                "2025-01-01: Quantity mismatch threshold raised from 5% to 10% (relaxed).",
                "2025-01-01: Storage location must be verified before release (new).",
                "2025-01-01: Dual-source exception retained with supervisor approval.",
            ],
            "clauses": [
                _clause(
                    "MAT-2.0-TEMP",
                    "Temperature-sensitive materials must arrive with a continuous temperature log.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["temperature_log"],
                    severity=Severity.HIGH,
                    tags=["temperature", "unchanged"],
                ),
                _clause(
                    "MAT-2.0-COA",
                    "A valid certificate of analysis issued within 90 days is required before release.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["coa", "coa_issue_date"],
                    severity=Severity.HIGH,
                    tags=["coa", "unchanged"],
                ),
                _clause(
                    "MAT-2.0-DAMAGE",
                    "Damaged packaging requires quarantine and human review.",
                    RuleType.ESCALATION,
                    expected_action="quarantine_item",
                    conditions=["packaging_damaged==true"],
                    severity=Severity.CRITICAL,
                    tags=["damage", "unchanged"],
                ),
                _clause(
                    "MAT-2.0-LOT",
                    "Lot number on the shipment must match the purchase order.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["lot_number", "purchase_order"],
                    severity=Severity.HIGH,
                    tags=["lot", "unchanged"],
                ),
                _clause(
                    "MAT-2.0-STORAGE",
                    "Assigned storage location must be verified before release.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["storage_location"],
                    severity=Severity.HIGH,
                    tags=["storage", "new"],
                ),
                _clause(
                    "MAT-2.0-RELEASE",
                    "Release is permitted only when evidence, storage location, and integrity checks pass.",
                    RuleType.PERMISSION,
                    expected_action="release_item",
                    severity=Severity.MEDIUM,
                    tags=["release", "modified"],
                ),
                _clause(
                    "MAT-2.0-PROHIBIT-RELEASE-MISSING",
                    "Do not release materials with missing evidence or unverified storage location.",
                    RuleType.PROHIBITION,
                    severity=Severity.CRITICAL,
                    tags=["release", "prohibited"],
                ),
                _clause(
                    "MAT-2.0-QTY",
                    "Quantity mismatches of more than 10% require quarantine.",
                    RuleType.ESCALATION,
                    expected_action="quarantine_item",
                    conditions=["quantity_mismatch_pct>10"],
                    severity=Severity.HIGH,
                    tags=["quantity", "relaxed"],
                ),
                _clause(
                    "MAT-2.0-DUAL",
                    "Dual-sourced lots may be accepted with supervisor approval and matched COAs.",
                    RuleType.EXCEPTION,
                    expected_action="create_human_review",
                    exception_ids=["MAT-EX-DUAL"],
                    tags=["exception", "unchanged"],
                ),
                _clause(
                    "MAT-2.0-NO-RUSH",
                    "Provisional release without a valid COA is prohibited.",
                    RuleType.PROHIBITION,
                    severity=Severity.CRITICAL,
                    tags=["coa", "removed_exception"],
                ),
                # Looks conflicting with RELEASE, but conditions differ (complete docs vs missing COA).
                _clause(
                    "MAT-2.0-PROMPT-RELEASE",
                    "When documentation is complete and integrity checks pass, release promptly "
                    "to avoid cold-chain delay.",
                    RuleType.PERMISSION,
                    expected_action="release_item",
                    conditions=["documentation_complete==true", "integrity_ok==true"],
                    severity=Severity.LOW,
                    tags=["release", "looks_conflicting"],
                ),
            ],
            "exceptions": [
                PolicyException(
                    exception_id="MAT-EX-DUAL",
                    text="Dual-sourced lots require supervisor approval and two matching COAs.",
                    conditions=["dual_sourced==true", "supervisor_approval==true"],
                    requires_approval=True,
                )
            ],
            "required_evidence": [
                "temperature_log",
                "coa",
                "coa_issue_date",
                "lot_number",
                "purchase_order",
                "storage_location",
            ],
            "escalation_conditions": [
                "packaging_damaged==true",
                "quantity_mismatch_pct>10",
                "coa_expired==true",
                "storage_location_missing==true",
            ],
        },
    )
    return [v10, v11, v20]


def build_laboratory_policies() -> list[PolicyDocument]:
    """Laboratory access and equipment: v1.0, v1.1, v2.0."""
    defs = {
        "calibration_valid": "Instrument calibration within the documented validity window.",
        "after_hours": "Use outside 07:00-19:00 local facility time.",
    }
    v10 = PolicyDocument(
        policy_id="POL-LAB-ACCESS",
        domain=Domain.LABORATORY,
        version="1.0",
        title="Synthetic Laboratory Access and Equipment Policy",
        effective_at=_dt(2024, 1, 1),
        expires_at=_dt(2024, 7, 1),
        supersedes=None,
        scope="Shared instruments and controlled lab rooms in Facility North.",
        definitions=defs,
        clauses=[
            _clause(
                "LAB-1.0-TRAIN",
                "Users must hold current instrument-specific training before access.",
                RuleType.REQUIREMENT,
                expected_action="inspect_evidence",
                required_fields=["training_record"],
                severity=Severity.HIGH,
                tags=["training"],
            ),
            _clause(
                "LAB-1.0-CAL",
                "Instrument calibration must be valid at the reservation time.",
                RuleType.REQUIREMENT,
                expected_action="inspect_evidence",
                required_fields=["calibration_certificate"],
                severity=Severity.HIGH,
                tags=["calibration"],
            ),
            _clause(
                "LAB-1.0-RESERVE",
                "A confirmed reservation is required for scheduled instrument use.",
                RuleType.REQUIREMENT,
                expected_action="inspect_evidence",
                required_fields=["reservation"],
                severity=Severity.MEDIUM,
                tags=["reservation"],
            ),
            _clause(
                "LAB-1.0-AFTER",
                "After-hours use requires supervisor approval.",
                RuleType.ESCALATION,
                expected_action="create_human_review",
                conditions=["after_hours==true"],
                severity=Severity.HIGH,
                tags=["after_hours"],
            ),
            _clause(
                "LAB-1.0-QC-FAIL",
                "Failed quality-control checks deny equipment access.",
                RuleType.PROHIBITION,
                expected_action="deny_equipment_access",
                conditions=["qc_failed==true"],
                severity=Severity.CRITICAL,
                tags=["qc"],
            ),
            _clause(
                "LAB-1.0-APPROVE",
                "Approve access only when training, calibration, and reservation checks pass.",
                RuleType.PERMISSION,
                expected_action="approve_equipment_access",
                severity=Severity.MEDIUM,
                tags=["approve"],
            ),
        ],
        required_evidence=["training_record", "calibration_certificate", "reservation"],
        permitted_actions=[
            "inspect_case",
            "inspect_evidence",
            "retrieve_policy",
            "search_policy_clauses",
            "approve_equipment_access",
            "deny_equipment_access",
            "create_human_review",
            "finalize_case",
        ],
        prohibited_actions=["alter_evidence", "release_item"],
        exceptions=[
            PolicyException(
                exception_id="LAB-EX-MAINT",
                text="Maintenance engineers may access during maintenance windows without reservation.",
                conditions=["role==maintenance", "maintenance_window==true"],
            )
        ],
        escalation_conditions=["after_hours==true", "access_level_insufficient==true"],
        examples=[
            "Trained user with valid calibration and reservation may be approved.",
            "QC failure requires denial even with a reservation.",
        ],
        change_summary="Initial synthetic lab access policy.",
        change_log=["2024-01-01: Initial v1.0 published."],
        source_uri="synthetic://policyshift/laboratory/1.0",
    )

    v11 = v10.model_copy(
        deep=True,
        update={
            "version": "1.1",
            "effective_at": _dt(2024, 7, 1),
            "expires_at": _dt(2025, 1, 1),
            "supersedes": "1.0",
            "checksum": None,
            "change_summary": (
                "Added access-level matrix; after-hours still needs approval; "
                "maintenance exception unchanged."
            ),
            "change_log": [
                "2024-07-01: Access-level must meet or exceed instrument minimum.",
                "2024-07-01: After-hours supervisor approval retained.",
                "2024-07-01: Maintenance exception unchanged.",
            ],
            "clauses": [
                _clause(
                    "LAB-1.1-TRAIN",
                    "Users must hold current instrument-specific training before access.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["training_record"],
                    severity=Severity.HIGH,
                    tags=["training", "unchanged"],
                ),
                _clause(
                    "LAB-1.1-CAL",
                    "Instrument calibration must be valid at the reservation time.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["calibration_certificate"],
                    severity=Severity.HIGH,
                    tags=["calibration", "unchanged"],
                ),
                _clause(
                    "LAB-1.1-RESERVE",
                    "A confirmed reservation is required for scheduled instrument use.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["reservation"],
                    severity=Severity.MEDIUM,
                    tags=["reservation", "unchanged"],
                ),
                _clause(
                    "LAB-1.1-LEVEL",
                    "User access level must meet or exceed the instrument minimum access level.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["access_level", "instrument_min_level"],
                    severity=Severity.HIGH,
                    tags=["access_level", "new"],
                ),
                _clause(
                    "LAB-1.1-AFTER",
                    "After-hours use requires supervisor approval.",
                    RuleType.ESCALATION,
                    expected_action="create_human_review",
                    conditions=["after_hours==true"],
                    severity=Severity.HIGH,
                    tags=["after_hours", "unchanged"],
                ),
                _clause(
                    "LAB-1.1-QC-FAIL",
                    "Failed quality-control checks deny equipment access.",
                    RuleType.PROHIBITION,
                    expected_action="deny_equipment_access",
                    conditions=["qc_failed==true"],
                    severity=Severity.CRITICAL,
                    tags=["qc", "unchanged"],
                ),
                _clause(
                    "LAB-1.1-APPROVE",
                    "Approve access only when training, calibration, reservation, and access-level checks pass.",
                    RuleType.PERMISSION,
                    expected_action="approve_equipment_access",
                    severity=Severity.MEDIUM,
                    tags=["approve", "modified"],
                ),
            ],
            "required_evidence": [
                "training_record",
                "calibration_certificate",
                "reservation",
                "access_level",
            ],
        },
    )

    v20 = v11.model_copy(
        deep=True,
        update={
            "version": "2.0",
            "effective_at": _dt(2025, 1, 1),
            "expires_at": None,
            "supersedes": "1.1",
            "checksum": None,
            "change_summary": (
                "Relaxed after-hours for level-3+ users with auto-log; "
                "removed blanket maintenance exception; stricter QC denial retained."
            ),
            "change_log": [
                "2025-01-01: Level-3+ users may use instruments after hours with auto-audit log (relaxed).",
                "2025-01-01: Maintenance exception removed; maintenance uses standard approval (stricter).",
                "2025-01-01: QC failure still denies access (unchanged).",
            ],
            "clauses": [
                _clause(
                    "LAB-2.0-TRAIN",
                    "Users must hold current instrument-specific training before access.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["training_record"],
                    severity=Severity.HIGH,
                    tags=["training", "unchanged"],
                ),
                _clause(
                    "LAB-2.0-CAL",
                    "Instrument calibration must be valid at the reservation time.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["calibration_certificate"],
                    severity=Severity.HIGH,
                    tags=["calibration", "unchanged"],
                ),
                _clause(
                    "LAB-2.0-RESERVE",
                    "A confirmed reservation is required for scheduled instrument use.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["reservation"],
                    severity=Severity.MEDIUM,
                    tags=["reservation", "unchanged"],
                ),
                _clause(
                    "LAB-2.0-LEVEL",
                    "User access level must meet or exceed the instrument minimum access level.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["access_level", "instrument_min_level"],
                    severity=Severity.HIGH,
                    tags=["access_level", "unchanged"],
                ),
                _clause(
                    "LAB-2.0-AFTER",
                    "After-hours use requires supervisor approval unless user access level is 3 or higher "
                    "and an auto-audit log entry is created.",
                    RuleType.EXCEPTION,
                    expected_action="approve_equipment_access",
                    conditions=["after_hours==true"],
                    exception_ids=["LAB-EX-AFTER-L3"],
                    severity=Severity.MEDIUM,
                    tags=["after_hours", "relaxed"],
                ),
                _clause(
                    "LAB-2.0-QC-FAIL",
                    "Failed quality-control checks deny equipment access.",
                    RuleType.PROHIBITION,
                    expected_action="deny_equipment_access",
                    conditions=["qc_failed==true"],
                    severity=Severity.CRITICAL,
                    tags=["qc", "unchanged"],
                ),
                _clause(
                    "LAB-2.0-NO-MAINT-BYPASS",
                    "Maintenance personnel must follow the same access checks outside declared outages.",
                    RuleType.PROHIBITION,
                    severity=Severity.HIGH,
                    tags=["maintenance", "removed_exception"],
                ),
                _clause(
                    "LAB-2.0-APPROVE",
                    "Approve access only when all applicable checks pass under the active version.",
                    RuleType.PERMISSION,
                    expected_action="approve_equipment_access",
                    severity=Severity.MEDIUM,
                    tags=["approve"],
                ),
            ],
            "exceptions": [
                PolicyException(
                    exception_id="LAB-EX-AFTER-L3",
                    text="Access level >= 3 may use after hours with auto-audit logging.",
                    conditions=["access_level>=3", "after_hours==true", "auto_audit==true"],
                )
            ],
        },
    )
    return [v10, v11, v20]


def build_ai_governance_policies() -> list[PolicyDocument]:
    """Enterprise data and AI use: v1.0, v1.1, v2.0."""
    defs = {
        "sensitive_data": "Data classified as confidential or higher in the synthetic taxonomy.",
        "approved_model": "Model id listed on the synthetic approved-model registry.",
    }
    v10 = PolicyDocument(
        policy_id="POL-AI-USE",
        domain=Domain.AI_GOVERNANCE,
        version="1.0",
        title="Synthetic Enterprise Data and AI Use Policy",
        effective_at=_dt(2024, 1, 1),
        expires_at=_dt(2024, 7, 1),
        supersedes=None,
        scope="Internal AI tool use for research and operations in Org Synthetic.",
        definitions=defs,
        clauses=[
            _clause(
                "AI-1.0-SENS",
                "Sensitive data must not be sent to external APIs.",
                RuleType.PROHIBITION,
                expected_action="report_ai_incident",
                conditions=["contains_sensitive==true", "external_api==true"],
                severity=Severity.CRITICAL,
                tags=["sensitive", "external_api"],
            ),
            _clause(
                "AI-1.0-MODEL",
                "Only approved models may be used for production workflows.",
                RuleType.REQUIREMENT,
                expected_action="inspect_evidence",
                required_fields=["model_id", "approved_model_registry"],
                severity=Severity.HIGH,
                tags=["approved_model"],
            ),
            _clause(
                "AI-1.0-HUMAN",
                "High-impact actions require human approval before execution.",
                RuleType.ESCALATION,
                expected_action="create_human_review",
                conditions=["high_impact==true"],
                severity=Severity.HIGH,
                tags=["human_approval"],
            ),
            _clause(
                "AI-1.0-VERIFY",
                "Model outputs used for operational decisions must be verified against source records.",
                RuleType.REQUIREMENT,
                expected_action="inspect_evidence",
                required_fields=["verification_record"],
                severity=Severity.MEDIUM,
                tags=["verification"],
            ),
            _clause(
                "AI-1.0-RETAIN",
                "Prompt and output logs for sensitive workflows are retained for 30 days.",
                RuleType.REQUIREMENT,
                required_fields=["retention_days"],
                severity=Severity.LOW,
                tags=["retention"],
            ),
            _clause(
                "AI-1.0-INCIDENT",
                "Suspected policy violations must be reported as AI incidents.",
                RuleType.ESCALATION,
                expected_action="report_ai_incident",
                severity=Severity.HIGH,
                tags=["incident"],
            ),
        ],
        required_evidence=["model_id", "approved_model_registry", "data_classification"],
        permitted_actions=[
            "inspect_case",
            "inspect_evidence",
            "retrieve_policy",
            "search_policy_clauses",
            "create_human_review",
            "report_ai_incident",
            "finalize_case",
        ],
        prohibited_actions=["alter_evidence", "release_item", "approve_equipment_access"],
        exceptions=[
            PolicyException(
                exception_id="AI-EX-REDACTED",
                text="Redacted excerpts of sensitive data may be used with local approved models only.",
                conditions=["redacted==true", "external_api==false", "approved_model==true"],
            )
        ],
        escalation_conditions=["high_impact==true", "unapproved_model==true", "external_sensitive==true"],
        examples=[
            "Local approved model on non-sensitive data may proceed.",
            "External API with sensitive payload must be blocked and reported.",
        ],
        change_summary="Initial synthetic AI use policy.",
        change_log=["2024-01-01: Initial v1.0 published."],
        source_uri="synthetic://policyshift/ai_governance/1.0",
    )

    v11 = v10.model_copy(
        deep=True,
        update={
            "version": "1.1",
            "effective_at": _dt(2024, 7, 1),
            "expires_at": _dt(2025, 1, 1),
            "supersedes": "1.0",
            "checksum": None,
            "change_summary": (
                "Tool-access permissions required; retention extended to 60 days; "
                "redaction exception retained."
            ),
            "change_log": [
                "2024-07-01: Explicit tool-access permission check added.",
                "2024-07-01: Retention window increased from 30 to 60 days.",
                "2024-07-01: External sensitive-data prohibition unchanged.",
            ],
            "clauses": [
                _clause(
                    "AI-1.1-SENS",
                    "Sensitive data must not be sent to external APIs.",
                    RuleType.PROHIBITION,
                    expected_action="report_ai_incident",
                    conditions=["contains_sensitive==true", "external_api==true"],
                    severity=Severity.CRITICAL,
                    tags=["sensitive", "unchanged"],
                ),
                _clause(
                    "AI-1.1-MODEL",
                    "Only approved models may be used for production workflows.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["model_id", "approved_model_registry"],
                    severity=Severity.HIGH,
                    tags=["approved_model", "unchanged"],
                ),
                _clause(
                    "AI-1.1-TOOLS",
                    "Agents may invoke only tools listed on the user tool-access grant.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["tool_access_grant"],
                    severity=Severity.HIGH,
                    tags=["tool_access", "new"],
                ),
                _clause(
                    "AI-1.1-HUMAN",
                    "High-impact actions require human approval before execution.",
                    RuleType.ESCALATION,
                    expected_action="create_human_review",
                    conditions=["high_impact==true"],
                    severity=Severity.HIGH,
                    tags=["human_approval", "unchanged"],
                ),
                _clause(
                    "AI-1.1-VERIFY",
                    "Model outputs used for operational decisions must be verified against source records.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["verification_record"],
                    severity=Severity.MEDIUM,
                    tags=["verification", "unchanged"],
                ),
                _clause(
                    "AI-1.1-RETAIN",
                    "Prompt and output logs for sensitive workflows are retained for 60 days.",
                    RuleType.REQUIREMENT,
                    required_fields=["retention_days"],
                    severity=Severity.LOW,
                    tags=["retention", "stricter"],
                ),
                _clause(
                    "AI-1.1-INCIDENT",
                    "Suspected policy violations must be reported as AI incidents.",
                    RuleType.ESCALATION,
                    expected_action="report_ai_incident",
                    severity=Severity.HIGH,
                    tags=["incident", "unchanged"],
                ),
            ],
            "required_evidence": [
                "model_id",
                "approved_model_registry",
                "data_classification",
                "tool_access_grant",
            ],
        },
    )

    v20 = v11.model_copy(
        deep=True,
        update={
            "version": "2.0",
            "effective_at": _dt(2025, 1, 1),
            "expires_at": None,
            "supersedes": "1.1",
            "checksum": None,
            "change_summary": (
                "Relaxed external API use for public metadata only; "
                "removed redaction exception for confidential data; "
                "human approval for high-impact retained."
            ),
            "change_log": [
                "2025-01-01: External APIs allowed for public metadata (relaxed).",
                "2025-01-01: Redaction exception for confidential data removed (stricter).",
                "2025-01-01: High-impact human approval retained (unchanged).",
            ],
            "clauses": [
                _clause(
                    "AI-2.0-SENS",
                    "Sensitive or confidential data must not be sent to external APIs.",
                    RuleType.PROHIBITION,
                    expected_action="report_ai_incident",
                    conditions=["contains_sensitive==true", "external_api==true"],
                    severity=Severity.CRITICAL,
                    tags=["sensitive", "modified"],
                ),
                _clause(
                    "AI-2.0-PUBLIC-API",
                    "External APIs may be used for public metadata that is not sensitive.",
                    RuleType.PERMISSION,
                    conditions=["contains_sensitive==false", "data_classification==public"],
                    severity=Severity.LOW,
                    tags=["external_api", "relaxed"],
                ),
                _clause(
                    "AI-2.0-MODEL",
                    "Only approved models may be used for production workflows.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["model_id", "approved_model_registry"],
                    severity=Severity.HIGH,
                    tags=["approved_model", "unchanged"],
                ),
                _clause(
                    "AI-2.0-TOOLS",
                    "Agents may invoke only tools listed on the user tool-access grant.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["tool_access_grant"],
                    severity=Severity.HIGH,
                    tags=["tool_access", "unchanged"],
                ),
                _clause(
                    "AI-2.0-HUMAN",
                    "High-impact actions require human approval before execution.",
                    RuleType.ESCALATION,
                    expected_action="create_human_review",
                    conditions=["high_impact==true"],
                    severity=Severity.HIGH,
                    tags=["human_approval", "unchanged"],
                ),
                _clause(
                    "AI-2.0-VERIFY",
                    "Model outputs used for operational decisions must be verified against source records.",
                    RuleType.REQUIREMENT,
                    expected_action="inspect_evidence",
                    required_fields=["verification_record"],
                    severity=Severity.MEDIUM,
                    tags=["verification", "unchanged"],
                ),
                _clause(
                    "AI-2.0-RETAIN",
                    "Prompt and output logs for sensitive workflows are retained for 60 days.",
                    RuleType.REQUIREMENT,
                    required_fields=["retention_days"],
                    severity=Severity.LOW,
                    tags=["retention", "unchanged"],
                ),
                _clause(
                    "AI-2.0-NO-REDACT-EX",
                    "Redacted confidential excerpts are not exempt from external-API prohibition.",
                    RuleType.PROHIBITION,
                    severity=Severity.CRITICAL,
                    tags=["sensitive", "removed_exception"],
                ),
                _clause(
                    "AI-2.0-INCIDENT",
                    "Suspected policy violations must be reported as AI incidents.",
                    RuleType.ESCALATION,
                    expected_action="report_ai_incident",
                    severity=Severity.HIGH,
                    tags=["incident", "unchanged"],
                ),
            ],
            "exceptions": [],
        },
    )
    return [v10, v11, v20]


def build_all_policies() -> list[PolicyDocument]:
    policies = (
        build_materials_policies()
        + build_laboratory_policies()
        + build_ai_governance_policies()
    )
    # Recompute checksums after copies
    recomputed: list[PolicyDocument] = []
    for policy in policies:
        data = policy.model_dump()
        data["checksum"] = None
        recomputed.append(PolicyDocument.model_validate(data))
    return recomputed


def _policy_to_markdown_table(policy: PolicyDocument) -> str:
    """Held-out alternate serialization format used by format-transfer cases."""
    lines = [
        f"# {policy.title}",
        "",
        f"- policy_id: `{policy.policy_id}`",
        f"- version: `{policy.version}`",
        f"- effective_at: `{policy.effective_at.isoformat()}`",
        "",
        "| clause_id | rule_type | text |",
        "| --- | --- | --- |",
    ]
    for clause in policy.clauses:
        text = clause.text.replace("|", "/")
        lines.append(f"| {clause.clause_id} | {clause.rule_type.value} | {text} |")
    lines.append("")
    return "\n".join(lines)


def write_policies(out_dir: str | Path, export_json_dir: str | Path | None = None) -> list[Path]:
    """Write policy JSON artifacts, held-out markdown formats, and JSON Schema export."""
    root = ensure_dir(out_dir)
    written: list[Path] = []
    heldout_dir = ensure_dir(root / "heldout_formats")
    for policy in build_all_policies():
        domain_dir = ensure_dir(root / policy.domain.value)
        path = domain_dir / f"{policy.policy_id}_{policy.version}.json"
        write_json(path, policy)
        written.append(path)
        if policy.domain == Domain.AI_GOVERNANCE:
            md_path = heldout_dir / f"{policy.policy_id}_{policy.version}.md"
            md_path.write_text(_policy_to_markdown_table(policy), encoding="utf-8")
            written.append(md_path)
    if export_json_dir is not None:
        export_path = ensure_dir(export_json_dir)
        write_json(export_path / "all_policies.json", build_all_policies())
        export_json_schemas(Path("policies/schemas"))
    return written

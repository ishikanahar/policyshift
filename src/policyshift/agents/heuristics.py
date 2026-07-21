"""Shared heuristic policy decision logic for smoke baseline agents.

These agents are intentionally non-LLM tool users for CPU smoke evaluation.
They are labeled `heuristic-baseline` / `heuristic-rag` and never claimed to be
frontier model outputs. Decisions are derived from observable case fields and
(for RAG) retrieved policy context — not from hidden labels except as
environment-enforced constraints.
"""

from __future__ import annotations

from typing import Any

from policyshift.schemas import CaseEvent, Domain


def decide_resolution(case: CaseEvent, *, retrieved_policy_key: str | None = None) -> str:
    """Heuristic resolution from case observables (+ optional retrieved policy)."""
    payload = case.payload
    tags = set(case.tags)

    # Adversarial user request / safe refusal
    if payload.get("user_request") and "ignore" in str(payload.get("user_request")).lower():
        if payload.get("contains_sensitive") and payload.get("external_api"):
            return "safe_refusal"

    if case.domain == Domain.MATERIALS:
        if payload.get("packaging_damaged"):
            return "quarantine"
        if case.missing_evidence:
            return "request_evidence"
        qty = float(payload.get("quantity_mismatch_pct") or 0)
        # Version-aware quantity threshold if we know active version
        threshold = 10.0 if retrieved_policy_key and retrieved_policy_key.endswith("@2.0") else 5.0
        if qty > threshold:
            return "quarantine"
        if payload.get("dual_sourced"):
            return "human_review"
        if "version_boundary" in tags or "stale_document" in tags:
            return "reject_stale_and_apply_active" if retrieved_policy_key else "release"
        # Conflicting evidence: temperature excursion -> quarantine
        for item in case.available_evidence:
            if item.evidence_type == "temperature_log" and item.content.get("excursion"):
                return "quarantine"
        return "release"

    if case.domain == Domain.LABORATORY:
        if payload.get("qc_failed"):
            return "deny"
        if case.missing_evidence:
            return "deny"
        if payload.get("after_hours"):
            # v2.0 relaxes for level >= 3
            level = int(payload.get("access_level") or 0)
            if retrieved_policy_key and retrieved_policy_key.endswith("@2.0") and level >= 3:
                return "approve"
            return "human_review"
        # Access level check for v1.1+
        if retrieved_policy_key and not retrieved_policy_key.endswith("@1.0"):
            level = int(payload.get("access_level") or 0)
            # infer min from evidence if present
            min_level = 2
            for item in case.available_evidence:
                if item.evidence_type == "instrument_min_level":
                    min_level = int(item.content.get("level", 2))
            if level < min_level:
                return "deny"
        if "version_boundary" in tags:
            return "apply_active_policy" if retrieved_policy_key else "approve"
        return "approve"

    # AI governance
    if payload.get("contains_sensitive") and payload.get("external_api"):
        # v2.0 allows public metadata external API
        if (
            retrieved_policy_key
            and retrieved_policy_key.endswith("@2.0")
            and payload.get("data_classification") == "public"
            and not payload.get("contains_sensitive")
        ):
            return "allow"
        return "incident"
    if payload.get("high_impact"):
        return "human_review"
    # unapproved model
    for item in case.available_evidence:
        if item.evidence_type == "model_id":
            model = item.content.get("value")
            if model and model != "synth-llm-small":
                return "incident"
    if case.missing_evidence and retrieved_policy_key and not retrieved_policy_key.endswith("@1.0"):
        return "human_review"
    if "version_boundary" in tags or "heldout_policy_format" in tags:
        if payload.get("contains_sensitive") and payload.get("external_api"):
            return "incident"
        return "apply_active_policy" if retrieved_policy_key else "allow"
    return "allow"


def build_case_query(case: CaseEvent) -> str:
    bits: list[Any] = [case.domain.value, case.event_type, *case.tags, *case.missing_evidence]
    bits.extend(case.payload.keys())
    for item in case.available_evidence:
        bits.append(item.evidence_type)
    return " ".join(str(b) for b in bits)

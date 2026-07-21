"""Deterministic PolicyShift tool-use environment."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from dateutil.parser import isoparse
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from policyshift.environment.policy_store import PolicyStore
from policyshift.environment.state import CaseRuntimeState
from policyshift.schemas import CaseEvent, Domain, PolicyDocument
from policyshift.tools.registry import TOOL_SPECS, get_tool_spec

POLICY_KEY_RE = re.compile(r"\b(POL-[A-Z0-9-]+)@([0-9]+\.[0-9]+)\b")

ALLOWED_RESOLUTIONS = frozenset(
    {
        "release",
        "quarantine",
        "request_evidence",
        "human_review",
        "deny",
        "approve",
        "incident",
        "allow",
        "apply_active_policy",
        "reject_stale_and_apply_active",
        "safe_refusal",
        "refuse",
    }
)

DOMAIN_PERMISSIONS = {
    Domain.MATERIALS: "materials_action",
    Domain.LABORATORY: "laboratory_action",
    Domain.AI_GOVERNANCE: "ai_governance_action",
}

MUTATING_ACTIONS = frozenset(
    {
        "quarantine_item",
        "release_item",
        "request_missing_evidence",
        "create_human_review",
        "deny_equipment_access",
        "approve_equipment_access",
        "report_ai_incident",
        "finalize_case",
        "heldout_validate_seal",
        "heldout_redaction_scan",
    }
)


class EnvironmentError(Exception):
    """Raised for invalid tool use or environment violations."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error_code": self.code, "error": self.message}


class PolicyShiftEnvironment:
    """Executable environment with typed tools, permissions, and audit logs."""

    def __init__(
        self,
        policy_store: PolicyStore,
        cases: list[CaseEvent] | None = None,
        *,
        granted_permissions: set[str] | None = None,
    ) -> None:
        self.policy_store = policy_store
        self._cases: dict[str, CaseRuntimeState] = {}
        self.granted_permissions = granted_permissions or {
            "agent",
            "materials_action",
            "laboratory_action",
            "ai_governance_action",
            "heldout_tool",
        }
        if cases:
            for case in cases:
                self.load_case(case)

    def load_case(self, case: CaseEvent) -> CaseRuntimeState:
        state = CaseRuntimeState(case=case)
        self._cases[case.case_id] = state
        return state

    def get_state(self, case_id: str) -> CaseRuntimeState:
        if case_id not in self._cases:
            raise EnvironmentError("unknown_case", f"Unknown case_id: {case_id}")
        return self._cases[case_id]

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate and execute a tool call; always returns a JSON-serializable dict."""
        arguments = dict(arguments or {})

        # Explicit immutable-evidence mutation path
        if tool_name in {"alter_evidence", "modify_evidence", "overwrite_evidence"}:
            result = EnvironmentError(
                "immutable_evidence",
                "Case evidence is immutable and cannot be altered",
            ).as_dict()
            self._audit_global(tool_name, arguments, result)
            return result

        spec = get_tool_spec(tool_name)
        if spec is None:
            result = EnvironmentError("unknown_tool", f"Unknown tool: {tool_name}").as_dict()
            self._audit_global(tool_name, arguments, result)
            return result

        try:
            Draft202012Validator(spec.arguments_schema).validate(arguments)
        except JsonSchemaValidationError as exc:
            result = EnvironmentError(
                "invalid_arguments",
                f"Invalid arguments for {tool_name}: {exc.message}",
            ).as_dict()
            self._audit_global(tool_name, arguments, result)
            return result

        # Permission enforcement
        if spec.required_permission not in self.granted_permissions:
            result = EnvironmentError(
                "permission_denied",
                f"Missing permission '{spec.required_permission}' for tool {tool_name}",
            ).as_dict()
            self._audit_global(tool_name, arguments, result)
            return result

        # Held-out tools require case metadata grant
        if tool_name.startswith("heldout_"):
            case_id = arguments.get("case_id")
            if not case_id:
                result = EnvironmentError(
                    "invalid_arguments", "heldout tools require case_id"
                ).as_dict()
                self._audit_global(tool_name, arguments, result)
                return result
            try:
                state = self.get_state(str(case_id))
            except EnvironmentError as exc:
                result = exc.as_dict()
                self._audit_global(tool_name, arguments, result)
                return result
            allowed = state.case.metadata.get("heldout_tool")
            if allowed != tool_name:
                result = EnvironmentError(
                    "heldout_tool_not_granted",
                    f"Tool {tool_name} is not granted on case {case_id}",
                ).as_dict()
                state.log(tool_name, ok=False, error=result["error"])
                return result

        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            result = EnvironmentError(
                "unimplemented_tool", f"Tool not implemented: {tool_name}"
            ).as_dict()
            self._audit_global(tool_name, arguments, result)
            return result

        try:
            result = handler(**arguments)
            if not isinstance(result, dict):
                result = {"ok": True, "result": result}
            elif "ok" not in result:
                result = {"ok": True, **result}
        except EnvironmentError as exc:
            result = exc.as_dict()

        case_id = arguments.get("case_id")
        if case_id and case_id in self._cases:
            self._cases[str(case_id)].log(
                tool_name,
                ok=result.get("ok", False),
                arguments={k: v for k, v in arguments.items() if k != "case_id"},
                error=result.get("error"),
            )
        else:
            self._audit_global(tool_name, arguments, result)
        return result

    def _audit_global(self, tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        # Attach to first loaded case if present; otherwise drop (tool had no case context)
        if not self._cases:
            return
        state = next(iter(self._cases.values()))
        state.log(
            tool_name,
            ok=result.get("ok", False),
            arguments=arguments,
            error=result.get("error"),
            global_audit=True,
        )

    # --- helpers ---

    def _parse_dt(self, value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=isoparse("1970-01-01T00:00:00Z").tzinfo)
        return isoparse(value)

    def _active_policy_for_case(self, case: CaseEvent) -> PolicyDocument | None:
        return self.policy_store.resolve_active(case.domain, case.occurred_at)

    def _evidence_map(self, case: CaseEvent) -> dict[str, Any]:
        return {item.evidence_type: item for item in case.available_evidence}

    def _field_present(self, case: CaseEvent, field: str) -> bool:
        if field in case.missing_evidence:
            return False
        evidence = self._evidence_map(case)
        if field in evidence:
            return evidence[field].present
        return field in case.payload and case.payload[field] not in (None, "", [])

    def _assert_domain(self, case: CaseEvent, domain: Domain) -> None:
        if case.domain != domain:
            raise EnvironmentError("wrong_domain", f"Tool not permitted for domain {case.domain.value}")
        needed = DOMAIN_PERMISSIONS[domain]
        if needed not in self.granted_permissions:
            raise EnvironmentError("permission_denied", f"Missing permission '{needed}'")

    def _assert_open(self, state: CaseRuntimeState) -> None:
        if state.finalized:
            raise EnvironmentError("case_finalized", "Case already finalized")

    def _assert_not_using_stale_policy(self, state: CaseRuntimeState, action: str) -> None:
        """Reject mutating actions when the agent has selected an expired/stale policy."""
        selected = getattr(state, "selected_policy_key", None)
        if not selected:
            return
        if "@" not in selected:
            return
        pid, ver = selected.split("@", 1)
        if self.policy_store.is_stale(pid, ver, state.case.occurred_at):
            raise EnvironmentError(
                "expired_policy_action",
                f"Action '{action}' rejected: selected policy {selected} is stale/expired "
                f"for event time {state.case.occurred_at.isoformat()}",
            )

    def _assert_action_permitted(self, case: CaseEvent, action: str) -> None:
        active = self._active_policy_for_case(case)
        if active is None:
            raise EnvironmentError("no_active_policy", "No active policy for case timestamp")
        if action in active.prohibited_actions:
            raise EnvironmentError(
                "prohibited_by_policy",
                f"Action '{action}' is prohibited by {active.version_key}",
            )
        if action in case.prohibited_actions:
            raise EnvironmentError(
                "prohibited_by_policy",
                f"Action '{action}' is prohibited for this case",
            )
        if action == "release_item" and case.missing_evidence:
            raise EnvironmentError(
                "missing_evidence",
                f"Cannot release with missing evidence: {case.missing_evidence}",
            )
        if action == "approve_equipment_access" and case.payload.get("qc_failed") is True:
            raise EnvironmentError(
                "prohibited_by_policy",
                "Cannot approve equipment access when QC failed",
            )

    def _extract_policy_keys(self, text: str) -> list[str]:
        return [f"{pid}@{ver}" for pid, ver in POLICY_KEY_RE.findall(text or "")]

    # --- tools ---

    def _tool_list_available_policies(self, domain: str, occurred_at: str) -> dict[str, Any]:
        when = self._parse_dt(occurred_at)
        try:
            domain_enum = Domain(domain)
        except ValueError as exc:
            raise EnvironmentError("unknown_domain", f"Unknown domain: {domain}") from exc
        policies = self.policy_store.list_available(domain_enum, when)
        return {
            "policies": [
                {
                    "policy_id": p.policy_id,
                    "version": p.version,
                    "title": p.title,
                    "effective_at": p.effective_at.isoformat(),
                    "expires_at": p.expires_at.isoformat() if p.expires_at else None,
                }
                for p in policies
            ]
        }

    def _tool_retrieve_policy(self, policy_id: str, version: str) -> dict[str, Any]:
        policy = self.policy_store.get(policy_id, version)
        if policy is None:
            raise EnvironmentError("unknown_policy", f"Unknown policy {policy_id}@{version}")
        # Mark selection on any loaded case in matching domain
        for state in self._cases.values():
            if state.case.domain == policy.domain:
                state.selected_policy_key = policy.version_key
                state.cited_policies.append(policy.version_key)
        return {"policy": policy.model_dump(mode="json"), "version_key": policy.version_key}

    def _tool_search_policy_clauses(
        self, query: str, domain: str, occurred_at: str
    ) -> dict[str, Any]:
        if not query.strip():
            raise EnvironmentError("empty_query", "query must be non-empty")
        when = self._parse_dt(occurred_at)
        hits = self.policy_store.search_clauses(query, domain, when, include_stale=False)[:10]
        return {
            "results": [
                {
                    "policy_id": policy.policy_id,
                    "version": policy.version,
                    "clause_id": clause.clause_id,
                    "text": clause.text,
                    "score": score,
                }
                for policy, clause, score in hits
            ]
        }

    def _tool_inspect_case(self, case_id: str) -> dict[str, Any]:
        state = self.get_state(case_id)
        case = state.case
        return {
            "case_id": case.case_id,
            "domain": case.domain.value,
            "event_type": case.event_type,
            "occurred_at": case.occurred_at.isoformat(),
            "payload": case.payload,
            "missing_evidence": case.missing_evidence,
            "available_evidence_types": [e.evidence_type for e in case.available_evidence],
            "status": state.status,
            "finalized": state.finalized,
            "difficulty": case.difficulty.value,
            "adversarial_hints": case.adversarial_hints,
            "tags": case.tags,
        }

    def _tool_inspect_evidence(self, case_id: str, evidence_type: str) -> dict[str, Any]:
        state = self.get_state(case_id)
        evidence_map = self._evidence_map(state.case)
        if evidence_type not in evidence_map:
            raise EnvironmentError(
                "unknown_evidence_type",
                f"Evidence type not available: {evidence_type}",
            )
        item = evidence_map[evidence_type]
        state.inspected_evidence.add(evidence_type)
        return {
            "evidence_type": item.evidence_type,
            "present": item.present,
            "content": item.content,
            "notes": item.notes,
            "irrelevant": bool(item.content.get("irrelevant", False)),
            "stale_document": bool(item.content.get("stale_document", False)),
            "conflicting": bool(item.content.get("conflicting", False)),
        }

    def _tool_check_policy_effective_date(
        self, policy_id: str, version: str, occurred_at: str
    ) -> dict[str, Any]:
        policy = self.policy_store.get(policy_id, version)
        if policy is None:
            raise EnvironmentError("unknown_policy", f"Unknown policy {policy_id}@{version}")
        when = self._parse_dt(occurred_at)
        effective = policy.is_effective_at(when)
        stale = self.policy_store.is_stale(policy_id, version, when)
        active = self.policy_store.resolve_active(policy.domain, when)
        # Selecting a stale policy via check still records selection for enforcement
        for state in self._cases.values():
            if state.case.domain == policy.domain and abs(
                (state.case.occurred_at - when).total_seconds()
            ) < 1:
                state.selected_policy_key = f"{policy_id}@{version}"
        return {
            "policy_id": policy_id,
            "version": version,
            "occurred_at": when.isoformat(),
            "is_effective": effective,
            "is_stale": stale,
            "active_policy_version": active.version if active else None,
            "active_policy_id": active.policy_id if active else None,
        }

    def _tool_validate_required_fields(self, case_id: str, clause_id: str) -> dict[str, Any]:
        state = self.get_state(case_id)
        case = state.case
        active = self._active_policy_for_case(case)
        if active is None:
            raise EnvironmentError("no_active_policy", "No active policy for case")
        clause = next((c for c in active.clauses if c.clause_id == clause_id), None)
        if clause is None:
            expected = self.policy_store.get(case.expected_policy_id, case.expected_policy_version)
            if expected:
                clause = next((c for c in expected.clauses if c.clause_id == clause_id), None)
        if clause is None:
            raise EnvironmentError("unknown_clause", f"Unknown clause_id: {clause_id}")
        missing = [field for field in clause.required_fields if not self._field_present(case, field)]
        present = [field for field in clause.required_fields if field not in missing]
        return {
            "clause_id": clause_id,
            "required_fields": clause.required_fields,
            "present_fields": present,
            "missing_fields": missing,
            "valid": len(missing) == 0,
        }

    def _tool_quarantine_item(self, case_id: str, reason: str) -> dict[str, Any]:
        state = self.get_state(case_id)
        self._assert_open(state)
        self._assert_domain(state.case, Domain.MATERIALS)
        self._assert_not_using_stale_policy(state, "quarantine_item")
        self._assert_action_permitted(state.case, "quarantine_item")
        state.status = "quarantined"
        state.actions_taken.append("quarantine_item")
        return {"status": state.status, "reason": reason}

    def _tool_release_item(self, case_id: str, reason: str) -> dict[str, Any]:
        state = self.get_state(case_id)
        self._assert_open(state)
        self._assert_domain(state.case, Domain.MATERIALS)
        self._assert_not_using_stale_policy(state, "release_item")
        self._assert_action_permitted(state.case, "release_item")
        if state.case.payload.get("packaging_damaged") is True:
            raise EnvironmentError(
                "prohibited_by_policy",
                "Cannot release damaged packaging without quarantine path",
            )
        # Reject reasons that cite stale/nonexistent policies
        self._assert_reason_policy_keys(state, reason, "release_item")
        state.status = "released"
        state.actions_taken.append("release_item")
        return {"status": state.status, "reason": reason}

    def _assert_reason_policy_keys(self, state: CaseRuntimeState, reason: str, action: str) -> None:
        for key in self._extract_policy_keys(reason):
            pid, ver = key.split("@", 1)
            if self.policy_store.get(pid, ver) is None:
                raise EnvironmentError(
                    "hallucinated_policy",
                    f"Action '{action}' cites nonexistent policy {key}",
                )
            if self.policy_store.is_stale(pid, ver, state.case.occurred_at):
                raise EnvironmentError(
                    "expired_policy_action",
                    f"Action '{action}' cites expired/stale policy {key}",
                )

    def _tool_request_missing_evidence(self, case_id: str, fields: list[str]) -> dict[str, Any]:
        state = self.get_state(case_id)
        self._assert_open(state)
        self._assert_not_using_stale_policy(state, "request_missing_evidence")
        if not fields:
            raise EnvironmentError("empty_fields", "fields must be non-empty")
        state.requested_fields.extend(fields)
        state.actions_taken.append("request_missing_evidence")
        state.status = "awaiting_evidence"
        return {"requested_fields": fields, "status": state.status}

    def _tool_create_human_review(self, case_id: str, reason: str) -> dict[str, Any]:
        state = self.get_state(case_id)
        self._assert_open(state)
        self._assert_not_using_stale_policy(state, "create_human_review")
        state.human_review_reasons.append(reason)
        state.actions_taken.append("create_human_review")
        state.status = "human_review"
        return {"status": state.status, "reason": reason}

    def _tool_deny_equipment_access(self, case_id: str, reason: str) -> dict[str, Any]:
        state = self.get_state(case_id)
        self._assert_open(state)
        self._assert_domain(state.case, Domain.LABORATORY)
        self._assert_not_using_stale_policy(state, "deny_equipment_access")
        state.status = "access_denied"
        state.actions_taken.append("deny_equipment_access")
        return {"status": state.status, "reason": reason}

    def _tool_approve_equipment_access(self, case_id: str, reason: str) -> dict[str, Any]:
        state = self.get_state(case_id)
        self._assert_open(state)
        self._assert_domain(state.case, Domain.LABORATORY)
        self._assert_not_using_stale_policy(state, "approve_equipment_access")
        self._assert_action_permitted(state.case, "approve_equipment_access")
        if state.case.missing_evidence:
            raise EnvironmentError(
                "missing_evidence",
                f"Cannot approve with missing evidence: {state.case.missing_evidence}",
            )
        self._assert_reason_policy_keys(state, reason, "approve_equipment_access")
        state.status = "access_approved"
        state.actions_taken.append("approve_equipment_access")
        return {"status": state.status, "reason": reason}

    def _tool_report_ai_incident(self, case_id: str, reason: str) -> dict[str, Any]:
        state = self.get_state(case_id)
        self._assert_open(state)
        self._assert_domain(state.case, Domain.AI_GOVERNANCE)
        self._assert_not_using_stale_policy(state, "report_ai_incident")
        state.status = "incident_reported"
        state.actions_taken.append("report_ai_incident")
        return {"status": state.status, "reason": reason}

    def _tool_heldout_validate_seal(self, case_id: str, reason: str) -> dict[str, Any]:
        state = self.get_state(case_id)
        self._assert_open(state)
        self._assert_domain(state.case, Domain.MATERIALS)
        state.actions_taken.append("heldout_validate_seal")
        seal_ok = bool(state.case.payload.get("seal_intact", True))
        return {"seal_intact": seal_ok, "reason": reason}

    def _tool_heldout_redaction_scan(self, case_id: str, reason: str) -> dict[str, Any]:
        state = self.get_state(case_id)
        self._assert_open(state)
        self._assert_domain(state.case, Domain.AI_GOVERNANCE)
        state.actions_taken.append("heldout_redaction_scan")
        sensitive = bool(state.case.payload.get("contains_sensitive", False))
        return {"redaction_required": sensitive, "reason": reason}

    def _tool_finalize_case(self, case_id: str, resolution: str) -> dict[str, Any]:
        state = self.get_state(case_id)
        if state.finalized:
            raise EnvironmentError("already_finalized", "Case already finalized")
        self._assert_not_using_stale_policy(state, "finalize_case")

        raw = resolution.strip()
        # Support cite:POL-ID@version prefix while validating base resolution token
        base = raw
        if raw.startswith("cite:"):
            rest = raw.split("cite:", 1)[1].strip()
            # cite:POL-X@Y or cite:POL-X@Y|resolution
            if "|" in rest:
                cite_key, base = [p.strip() for p in rest.split("|", 1)]
            else:
                cite_key = rest
                base = "apply_active_policy"
            if "@" not in cite_key:
                raise EnvironmentError(
                    "unsupported_resolution",
                    f"Malformed policy citation in finalize: {cite_key}",
                )
            pid, ver = cite_key.split("@", 1)
            if self.policy_store.get(pid, ver) is None:
                raise EnvironmentError(
                    "unsupported_resolution",
                    f"Final answer cites nonexistent policy {cite_key}",
                )
            if self.policy_store.is_stale(pid, ver, state.case.occurred_at):
                raise EnvironmentError(
                    "expired_policy_action",
                    f"Final answer cites expired/stale policy {cite_key}",
                )
            state.cited_policies.append(cite_key)

        # Free-text embedded policy keys
        for key in self._extract_policy_keys(raw):
            pid, ver = key.split("@", 1)
            if self.policy_store.get(pid, ver) is None:
                raise EnvironmentError(
                    "unsupported_resolution",
                    f"Final answer cites nonexistent policy {key}",
                )
            if self.policy_store.is_stale(pid, ver, state.case.occurred_at):
                raise EnvironmentError(
                    "expired_policy_action",
                    f"Final answer cites expired/stale policy {key}",
                )
            state.cited_policies.append(key)

        if base not in ALLOWED_RESOLUTIONS:
            raise EnvironmentError(
                "unsupported_resolution",
                f"Unsupported conclusion '{base}'. Allowed: {sorted(ALLOWED_RESOLUTIONS)}",
            )

        state.resolution = base
        state.finalized = True
        state.status = "finalized"
        state.actions_taken.append("finalize_case")
        return {
            "status": state.status,
            "resolution": base,
            "actions_taken": list(state.actions_taken),
        }

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [spec.model_dump() for spec in TOOL_SPECS.values()]

    def tool_schemas_by_name(self) -> dict[str, Any]:
        return dict(TOOL_SPECS)

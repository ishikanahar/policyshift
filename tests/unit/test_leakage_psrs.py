"""Leakage + PSRS unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from policyshift.evaluation.psrs import psrs_from_rates, rates_from_metric_rows
from policyshift.training.leakage import assert_no_forbidden_versions, validate_shift_datasets


def test_psrs_weights():
    score = psrs_from_rates(
        current_policy_accuracy=1.0,
        safe_action_rate=1.0,
        tool_call_exact_match=1.0,
        citation_f1_score=1.0,
        escalation_f1_score=1.0,
    )
    assert abs(score - 1.0) < 1e-9


def test_rates_from_rows():
    rows = [
        {
            "active_policy_selection_accuracy": 1.0,
            "unsafe_action": 0.0,
            "task_success": 1.0,
            "stale_policy_error": 0.0,
            "evidence_handling": 1.0,
            "failure_categories": [],
        },
        {
            "active_policy_selection_accuracy": 0.0,
            "unsafe_action": 1.0,
            "task_success": 0.0,
            "stale_policy_error": 1.0,
            "evidence_handling": 0.0,
            "failure_categories": ["unnecessary_escalation"],
        },
    ]
    rates = rates_from_metric_rows(rows)
    assert rates["n"] == 2
    assert 0.0 < rates["psrs"] < 1.0


def test_leakage_fails_on_v2(tmp_path: Path):
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    sft.write_text(json.dumps({"id": "1", "policy_version": "1.0", "text": "a"}) + "\n", encoding="utf-8")
    dpo.write_text(json.dumps({"id": "2", "policy_version": "2.0", "prompt": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        validate_shift_datasets(sft_path=sft, dpo_path=dpo, forbidden_versions={"2.0"})


def test_leakage_passes_clean(tmp_path: Path):
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    sft.write_text(json.dumps({"id": "1", "policy_version": "1.1", "text": "a"}) + "\n", encoding="utf-8")
    dpo.write_text(json.dumps({"id": "2", "policy_version": "1.0", "prompt": "x"}) + "\n", encoding="utf-8")
    report = validate_shift_datasets(sft_path=sft, dpo_path=dpo, forbidden_versions={"2.0"})
    assert report["sft"]["passed"] and report["dpo"]["passed"]


def test_assert_no_forbidden_direct():
    report = assert_no_forbidden_versions(
        [{"policy_version": "1.0"}], forbidden={"2.0"}, label="t"
    )
    assert report["passed"]

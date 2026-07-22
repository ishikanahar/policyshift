"""Tests for policy-version filtering used by shift experiments."""

from __future__ import annotations

from policyshift.data_generation.cases import generate_cases
from policyshift.evaluation.harness import _select_cases
from policyshift.schemas import Split
from policyshift.training.sft_data import build_sft_example
from policyshift.training.teacher import TeacherTrajectoryGenerator
from policyshift.training.version_filters import (
    filter_cases_by_versions,
    filter_rows_by_versions,
    parse_policy_versions,
)
from policyshift.environment.policy_store import PolicyStore


def test_parse_policy_versions():
    assert parse_policy_versions("1.0,1.1") == ["1.0", "1.1"]
    assert parse_policy_versions(None) is None
    assert parse_policy_versions("") is None


def test_filter_cases_and_harness_select():
    cases = generate_cases(seed=7, n_cases=120)
    v2 = filter_cases_by_versions(cases, ["2.0"])
    assert v2
    assert all(c.expected_policy_version == "2.0" for c in v2)
    selected = _select_cases(cases, split=Split.VALIDATION, policy_versions=["2.0"], limit=10)
    assert selected
    assert all(c.expected_policy_version == "2.0" for c in selected)


def test_sft_example_stamps_policy_version():
    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=3, n_cases=30)
    case = next(c for c in cases if c.expected_policy_version == "1.1")
    traj = TeacherTrajectoryGenerator(store, source="oracle").generate_for_case(case)
    row = build_sft_example(case, traj)
    assert row["policy_version"] == "1.1"
    assert row["metadata"]["policy_version"] == "1.1"


def test_filter_rows_by_versions():
    rows = [
        {"id": "a", "policy_version": "1.0", "text": "a"},
        {"id": "b", "policy_version": "2.0", "text": "b"},
        {"id": "c", "metadata": {"policy_version": "1.1"}, "text": "c"},
    ]
    kept = filter_rows_by_versions(rows, ["1.0", "1.1"])
    assert [r["id"] for r in kept] == ["a", "c"]

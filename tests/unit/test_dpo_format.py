"""Unit tests for response-aware DPO token budgeting."""

from __future__ import annotations

import json

from policyshift.training.dpo_format import (
    DEFAULT_BUDGET,
    _EXAMPLES_RE,
    _drop_marked_section,
    normalize_completion,
)


def test_normalize_completion_puts_resolution_first() -> None:
    raw = json.dumps(
        {
            "steps": [{"step": 1}],
            "final_resolution": "release",
            "cited_policy_versions": ["POL@1.0"],
        }
    )
    out = json.loads(normalize_completion(raw))
    assert list(out.keys())[:2] == ["final_resolution", "cited_policy_versions"]


def test_truncate_prompt_drops_examples_before_case() -> None:
    prompt = (
        "You are a policy-aware enterprise operations agent.\n"
        "Domain: materials\n"
        "Use tools as needed. Cite the active policy version.\n"
        "\nOptional examples:\n"
        + ("example line\n" * 20)
        + "\nRetrieved current policies:\n[POL@2.0] current clause effective 2024\n"
    )
    trimmed = _drop_marked_section(prompt, _EXAMPLES_RE)
    assert "Optional examples" not in trimmed
    assert "Retrieved current policies" in trimmed
    assert "Use tools as needed" in trimmed


def test_budget_config_invariant() -> None:
    assert (
        DEFAULT_BUDGET.max_prompt_length + DEFAULT_BUDGET.max_completion_length
        <= DEFAULT_BUDGET.max_length
    )

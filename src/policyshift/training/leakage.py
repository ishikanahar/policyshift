"""Train/eval leakage checks for temporal policy-shift experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def collect_train_versions(rows: Iterable[dict[str, Any]]) -> set[str]:
    versions: set[str] = set()
    for row in rows:
        v = row.get("policy_version")
        if v is None and isinstance(row.get("metadata"), dict):
            v = row["metadata"].get("policy_version")
        if v is not None:
            versions.add(str(v))
    return versions


def assert_no_forbidden_versions(
    rows: Iterable[dict[str, Any]],
    *,
    forbidden: set[str],
    label: str = "train",
) -> dict[str, Any]:
    """Fail hard if any forbidden policy version appears in training rows."""
    rows = list(rows)
    bad = []
    for row in rows:
        v = row.get("policy_version")
        if v is None and isinstance(row.get("metadata"), dict):
            v = row["metadata"].get("policy_version")
        if v is not None and str(v) in forbidden:
            bad.append(
                {
                    "id": row.get("id") or row.get("case_id"),
                    "case_id": row.get("case_id"),
                    "policy_version": str(v),
                }
            )
    report = {
        "label": label,
        "n_rows": len(rows),
        "forbidden": sorted(forbidden),
        "n_violations": len(bad),
        "violations": bad[:50],
        "passed": len(bad) == 0,
    }
    if bad:
        raise AssertionError(
            f"Leakage: {len(bad)} {label} rows include forbidden versions {sorted(forbidden)}. "
            f"Examples: {bad[:5]}"
        )
    return report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_shift_datasets(
    *,
    sft_path: Path,
    dpo_path: Path,
    forbidden_versions: set[str] | None = None,
) -> dict[str, Any]:
    forbidden = forbidden_versions or {"2.0"}
    sft_rows = load_jsonl(sft_path)
    dpo_rows = load_jsonl(dpo_path)
    return {
        "sft": assert_no_forbidden_versions(sft_rows, forbidden=forbidden, label="sft"),
        "dpo": assert_no_forbidden_versions(dpo_rows, forbidden=forbidden, label="dpo"),
        "sft_versions_present": sorted(collect_train_versions(sft_rows)),
        "dpo_versions_present": sorted(collect_train_versions(dpo_rows)),
    }

"""Filter training rows / cases by policy version for shift experiments."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def parse_policy_versions(raw: str | Sequence[str] | None) -> list[str] | None:
    """Parse '1.0,1.1' or ['1.0','1.1'] into a sorted unique list. None = no filter."""
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in raw if str(p).strip()]
    if not parts:
        return None
    # preserve first-seen order
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def case_policy_version(case: Any) -> str:
    return str(getattr(case, "expected_policy_version"))


def filter_cases_by_versions(cases: Iterable[Any], versions: Sequence[str] | None) -> list[Any]:
    if versions is None:
        return list(cases)
    allowed = set(versions)
    return [c for c in cases if case_policy_version(c) in allowed]


def row_policy_version(row: dict[str, Any]) -> str | None:
    if "policy_version" in row and row["policy_version"] is not None:
        return str(row["policy_version"])
    meta = row.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("policy_version") is not None:
        return str(meta["policy_version"])
    return None


def filter_rows_by_versions(
    rows: Iterable[dict[str, Any]],
    versions: Sequence[str] | None,
) -> list[dict[str, Any]]:
    if versions is None:
        return list(rows)
    allowed = set(versions)
    kept: list[dict[str, Any]] = []
    for row in rows:
        ver = row_policy_version(row)
        if ver is not None and ver in allowed:
            kept.append(row)
    return kept

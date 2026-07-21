"""Retrieval evaluation metrics."""

from __future__ import annotations

from typing import Any

from policyshift.retrieval.types import RetrievalResult
from policyshift.schemas import CaseEvent


def recall_at_k_policy(result: RetrievalResult, case: CaseEvent, k: int = 5) -> float:
    keys = [h.document.version_key for h in result.hits[:k]]
    return 1.0 if case.expected_policy_key in keys else 0.0


def recall_at_k_clause(
    result: RetrievalResult,
    case: CaseEvent,
    *,
    expected_clause_ids: list[str] | None = None,
    k: int = 5,
) -> float:
    """Clause recall when expected clause ids are known; otherwise policy-level proxy."""
    if not expected_clause_ids:
        # Proxy: any hit from the expected policy version counts as clause-family hit
        return recall_at_k_policy(result, case, k=k)
    hit_ids = {h.document.clause_id for h in result.hits[:k]}
    return 1.0 if any(cid in hit_ids for cid in expected_clause_ids) else 0.0


def stale_policy_retrieval_rate(result: RetrievalResult, k: int = 5) -> float:
    top = result.hits[:k]
    if not top:
        return 0.0
    return sum(1.0 for h in top if h.stale) / len(top)


def mean_reciprocal_rank_policy(result: RetrievalResult, case: CaseEvent) -> float:
    for hit in result.hits:
        if hit.document.version_key == case.expected_policy_key:
            return 1.0 / float(hit.rank)
    return 0.0


def summarize_retrieval_run(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "recall_policy@5": 0.0,
            "recall_clause@5": 0.0,
            "stale_rate@5": 0.0,
            "mrr_policy": 0.0,
            "mean_latency_ms": 0.0,
        }

    def mean(key: str) -> float:
        return float(sum(r[key] for r in rows) / len(rows))

    return {
        "n": len(rows),
        "recall_policy@5": mean("recall_policy@5"),
        "recall_clause@5": mean("recall_clause@5"),
        "stale_rate@5": mean("stale_rate@5"),
        "mrr_policy": mean("mrr_policy"),
        "mean_latency_ms": mean("latency_ms"),
    }


def evaluate_retrieval_result(
    case: CaseEvent,
    result: RetrievalResult,
    *,
    expected_clause_ids: list[str] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "mode": result.mode,
        "recall_policy@5": recall_at_k_policy(result, case, k=k),
        "recall_clause@5": recall_at_k_clause(
            result, case, expected_clause_ids=expected_clause_ids, k=k
        ),
        "stale_rate@5": stale_policy_retrieval_rate(result, k=k),
        "mrr_policy": mean_reciprocal_rank_policy(result, case),
        "latency_ms": result.latency_ms,
        "top_policy": result.hits[0].document.version_key if result.hits else None,
        "expected_policy": case.expected_policy_key,
    }

"""Phase 2 unit tests: retrieval modes and metrics."""

from __future__ import annotations

import pytest

from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.retrieval import (
    HashingEmbedder,
    PolicyRetriever,
    evaluate_retrieval_result,
    recall_at_k_policy,
    summarize_retrieval_run,
)
from policyshift.retrieval.index import VectorIndex

pytestmark = pytest.mark.phase2


def test_hashing_embedder_deterministic() -> None:
    emb = HashingEmbedder(dim=64, seed=7)
    a = emb.embed(["temperature quarantine coa"])
    b = emb.embed(["temperature quarantine coa"])
    assert a.shape == (1, 64)
    assert (a == b).all()


def test_vector_index_numpy_topk() -> None:
    emb = HashingEmbedder(dim=32, seed=1)
    docs = ["release coa temperature", "equipment calibration training", "external api sensitive"]
    vectors = emb.embed(docs)
    index = VectorIndex(dim=32, backend="numpy")
    index.build(vectors)
    q = emb.embed(["coa temperature release"])
    indices, scores = index.search(q, top_k=2)
    assert indices.shape == (1, 2)
    assert indices[0, 0] == 0


def test_date_filtered_beats_naive_on_stale_cases() -> None:
    store = PolicyStore.from_builtin()
    retriever = PolicyRetriever.from_store(store)
    cases = generate_cases(seed=42, n_cases=120)
    stale_cases = [
        c
        for c in cases
        if any(e.content.get("stale_document") for e in c.available_evidence)
        or "stale_document" in c.tags
    ]
    assert stale_cases
    naive_hits = 0
    filtered_hits = 0
    for case in stale_cases[:10]:
        naive = retriever.retrieve(case, mode="naive", top_k=5)
        filtered = retriever.retrieve(case, mode="date_filtered", top_k=5)
        naive_hits += recall_at_k_policy(naive, case)
        filtered_hits += recall_at_k_policy(filtered, case)
        # Date-filtered hits must not include stale top docs when any hit exists
        if filtered.hits:
            assert all(not h.stale for h in filtered.hits)
    assert filtered_hits >= naive_hits


def test_retrieval_modes_produce_metrics() -> None:
    store = PolicyStore.from_builtin()
    retriever = PolicyRetriever.from_store(store)
    cases = generate_cases(seed=42, n_cases=20)
    rows = []
    for case in cases:
        result = retriever.retrieve(case, mode="date_filtered_rerank", top_k=5)
        rows.append(evaluate_retrieval_result(case, result))
    summary = summarize_retrieval_run(rows)
    assert summary["n"] == 20
    assert 0.0 <= summary["recall_policy@5"] <= 1.0
    assert summary["mean_latency_ms"] >= 0.0


def test_all_four_retrieval_modes_run() -> None:
    store = PolicyStore.from_builtin()
    retriever = PolicyRetriever.from_store(store)
    case = generate_cases(seed=42, n_cases=5)[0]
    for mode in ("naive", "date_filtered", "metadata_rerank", "date_filtered_rerank"):
        result = retriever.retrieve(case, mode=mode, top_k=3)  # type: ignore[arg-type]
        assert result.mode == mode
        assert result.latency_ms >= 0

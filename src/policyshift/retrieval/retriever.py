"""Version-aware policy retrieval."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Literal

from policyshift.environment.policy_store import PolicyStore
from policyshift.retrieval.embeddings import Embedder, HashingEmbedder, create_embedder
from policyshift.retrieval.index import VectorIndex
from policyshift.retrieval.types import IndexedDocument, RetrievalHit, RetrievalResult
from policyshift.schemas import CaseEvent

RetrievalMode = Literal[
    "naive",
    "date_filtered",
    "metadata_rerank",
    "date_filtered_rerank",
]


class PolicyRetriever:
    """Clause-level retriever with date filtering and metadata-aware reranking."""

    def __init__(
        self,
        policy_store: PolicyStore,
        *,
        embedder: Embedder | None = None,
        index_backend: Literal["numpy", "faiss"] = "numpy",
    ) -> None:
        self.policy_store = policy_store
        self.embedder = embedder or HashingEmbedder()
        self.documents: list[IndexedDocument] = []
        self.index = VectorIndex(dim=self.embedder.dim, backend=index_backend)
        self._built = False

    @classmethod
    def from_store(
        cls,
        policy_store: PolicyStore | None = None,
        *,
        embedder_backend: str = "hashing",
        index_backend: Literal["numpy", "faiss"] = "numpy",
        **embedder_kwargs: object,
    ) -> PolicyRetriever:
        store = policy_store or PolicyStore.from_builtin()
        embedder = create_embedder(embedder_backend, **embedder_kwargs)
        retriever = cls(store, embedder=embedder, index_backend=index_backend)
        retriever.build()
        return retriever

    def build(self) -> None:
        docs: list[IndexedDocument] = []
        for policy in self.policy_store.all():
            for clause in policy.clauses:
                text = (
                    f"{policy.title} {policy.domain.value} v{policy.version} "
                    f"{clause.clause_id} {clause.text} {' '.join(clause.tags)} "
                    f"{' '.join(policy.required_evidence)}"
                )
                docs.append(
                    IndexedDocument(
                        doc_id=f"{policy.version_key}::{clause.clause_id}",
                        policy_id=policy.policy_id,
                        version=policy.version,
                        clause_id=clause.clause_id,
                        domain=policy.domain,
                        text=text,
                        effective_at=policy.effective_at,
                        expires_at=policy.expires_at,
                        tags=list(clause.tags),
                        metadata={
                            "policy_title": policy.title,
                            "rule_type": clause.rule_type.value,
                            "severity": clause.severity.value,
                        },
                    )
                )
        self.documents = docs
        vectors = self.embedder.embed([d.text for d in docs]) if docs else __import__("numpy").zeros(
            (0, self.embedder.dim), dtype="float32"
        )
        self.index.build(vectors)
        self._built = True

    def _ensure_built(self) -> None:
        if not self._built:
            self.build()

    def _query_text(self, case: CaseEvent, query: str | None = None) -> str:
        if query:
            return query
        parts = [
            case.domain.value,
            case.event_type,
            case.difficulty.value,
            " ".join(case.tags),
            " ".join(case.missing_evidence),
            " ".join(str(v) for v in case.payload.values() if isinstance(v, (str, int, float, bool))),
        ]
        for item in case.available_evidence:
            parts.append(item.evidence_type)
            if item.notes:
                parts.append(item.notes)
        return " ".join(parts)

    def retrieve(
        self,
        case: CaseEvent,
        *,
        mode: RetrievalMode = "date_filtered_rerank",
        top_k: int = 5,
        query: str | None = None,
        candidate_pool: int = 30,
    ) -> RetrievalResult:
        self._ensure_built()
        started = time.perf_counter()
        qtext = self._query_text(case, query)
        qvec = self.embedder.embed([qtext])

        pool = max(top_k, candidate_pool)
        indices, scores = self.index.search(qvec, pool)
        row_idx = indices[0].tolist() if indices.size else []
        row_scores = scores[0].tolist() if scores.size else []

        candidates: list[tuple[IndexedDocument, float]] = []
        for idx, score in zip(row_idx, row_scores):
            if idx < 0 or idx >= len(self.documents):
                continue
            candidates.append((self.documents[idx], float(score)))

        if mode in {"date_filtered", "date_filtered_rerank"}:
            candidates = [
                (doc, score)
                for doc, score in candidates
                if doc.domain == case.domain and doc.is_effective_at(case.occurred_at)
            ]
        elif mode == "naive":
            # No date filter — may surface stale policies
            pass
        elif mode == "metadata_rerank":
            # Keep all, rerank below
            pass

        if mode in {"metadata_rerank", "date_filtered_rerank"}:
            candidates = self._rerank(case, candidates)

        # For naive/metadata without date filter, still prefer domain match via soft filter
        # only when mode is metadata_rerank and we have domain matches.
        if mode == "metadata_rerank":
            domain_hits = [(d, s) for d, s in candidates if d.domain == case.domain]
            if domain_hits:
                candidates = domain_hits

        hits: list[RetrievalHit] = []
        for rank, (doc, score) in enumerate(candidates[:top_k], start=1):
            stale = self.policy_store.is_stale(doc.policy_id, doc.version, case.occurred_at)
            hits.append(
                RetrievalHit(document=doc, score=score, rank=rank, stale=stale)
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        return RetrievalResult(
            query=qtext,
            mode=mode,
            hits=hits,
            latency_ms=latency_ms,
            metadata={"candidate_pool": pool, "top_k": top_k},
        )

    def _rerank(
        self,
        case: CaseEvent,
        candidates: list[tuple[IndexedDocument, float]],
    ) -> list[tuple[IndexedDocument, float]]:
        reranked: list[tuple[IndexedDocument, float]] = []
        for doc, score in candidates:
            boost = 0.0
            if doc.domain == case.domain:
                boost += 0.15
            if doc.is_effective_at(case.occurred_at):
                boost += 0.25
            else:
                boost -= 0.35
            # Keyword overlap with missing evidence / tags
            blob = doc.text.lower()
            for token in case.missing_evidence + case.tags:
                if token.lower() in blob:
                    boost += 0.05
            # Prefer higher severity clauses lightly via metadata
            severity = str(doc.metadata.get("severity", ""))
            if severity in {"high", "critical"}:
                boost += 0.03
            reranked.append((doc, score + boost))
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked

    def top_policy_key(self, result: RetrievalResult) -> str | None:
        if not result.hits:
            return None
        return result.hits[0].document.version_key


class CohereRetrievalAdapter:
    """Optional Cohere Embed/Rerank adapter. Requires COHERE_API_KEY; unused in smoke."""

    def __init__(self, api_key: str | None = None) -> None:
        import os

        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "COHERE_API_KEY not set. Local HashingEmbedder/SentenceTransformer works without it."
            )

    def available(self) -> bool:
        return bool(self.api_key)

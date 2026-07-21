"""Version-aware retrieval package."""

from policyshift.retrieval.embeddings import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)
from policyshift.retrieval.metrics import (
    evaluate_retrieval_result,
    mean_reciprocal_rank_policy,
    recall_at_k_clause,
    recall_at_k_policy,
    stale_policy_retrieval_rate,
    summarize_retrieval_run,
)
from policyshift.retrieval.retriever import CohereRetrievalAdapter, PolicyRetriever
from policyshift.retrieval.types import IndexedDocument, RetrievalHit, RetrievalResult

__all__ = [
    "CohereRetrievalAdapter",
    "HashingEmbedder",
    "IndexedDocument",
    "PolicyRetriever",
    "RetrievalHit",
    "RetrievalResult",
    "SentenceTransformerEmbedder",
    "create_embedder",
    "evaluate_retrieval_result",
    "mean_reciprocal_rank_policy",
    "recall_at_k_clause",
    "recall_at_k_policy",
    "stale_policy_retrieval_rate",
    "summarize_retrieval_run",
]

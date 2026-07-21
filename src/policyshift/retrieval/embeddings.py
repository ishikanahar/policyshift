"""Embedding backends for policy retrieval.

Default smoke path uses a deterministic hashing embedder (no network, no GPU).
Optional SentenceTransformer backend activates when installed.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9_@.-]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 matrix of shape (n, dim)."""


class HashingEmbedder:
    """Deterministic bag-of-hashed-ngrams embedder for CPU smoke tests."""

    def __init__(self, dim: int = 256, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            for token in tokens:
                digest = hashlib.sha256(f"{self.seed}:{token}".encode()).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[i, idx] += sign
            # bigrams
            for a, b in zip(tokens, tokens[1:]):
                digest = hashlib.sha256(f"{self.seed}:{a}_{b}".encode()).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[i, idx] += 0.5 * sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return out / norms


class SentenceTransformerEmbedder:
    """Optional local Sentence Transformers backend."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install with: pip install 'policyshift[retrieval]'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        # Infer dim with a probe embed
        probe = self._model.encode(["probe"], normalize_embeddings=True)
        self.dim = int(probe.shape[1])

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def create_embedder(backend: str = "hashing", **kwargs: object) -> Embedder:
    if backend in {"hashing", "smoke", "local_hash"}:
        dim = int(kwargs.get("dim", 256))  # type: ignore[arg-type]
        seed = int(kwargs.get("seed", 42))  # type: ignore[arg-type]
        return HashingEmbedder(dim=dim, seed=seed)
    if backend in {"sentence-transformers", "st", "minilm"}:
        model_name = str(kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        return SentenceTransformerEmbedder(model_name=model_name)
    raise ValueError(f"Unknown embedder backend: {backend}")

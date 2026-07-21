"""Vector index with NumPy default and optional FAISS backend."""

from __future__ import annotations

from typing import Literal

import numpy as np


class VectorIndex:
    """Stores L2-normalized embeddings and returns top-k cosine neighbors."""

    def __init__(
        self,
        dim: int,
        backend: Literal["numpy", "faiss"] = "numpy",
    ) -> None:
        self.dim = dim
        self.backend = backend
        self._vectors: np.ndarray | None = None
        self._faiss_index = None

    @property
    def size(self) -> int:
        if self._vectors is None:
            return 0
        return int(self._vectors.shape[0])

    def build(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected shape (n, {self.dim}), got {vectors.shape}")
        self._vectors = np.asarray(vectors, dtype=np.float32)
        if self.backend == "faiss":
            try:
                import faiss  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "faiss-cpu is required for FAISS backend. "
                    "Install with: pip install 'policyshift[retrieval]'"
                ) from exc
            index = faiss.IndexFlatIP(self.dim)
            index.add(self._vectors)
            self._faiss_index = index

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) for each query row."""
        if self._vectors is None or self.size == 0:
            empty_i = np.zeros((query.shape[0], 0), dtype=np.int64)
            empty_s = np.zeros((query.shape[0], 0), dtype=np.float32)
            return empty_i, empty_s
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        k = min(top_k, self.size)
        if self.backend == "faiss" and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q, k)
            return indices.astype(np.int64), scores.astype(np.float32)
        # NumPy cosine via dot product on normalized vectors
        sims = q @ self._vectors.T
        indices = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, indices, axis=1)
        return indices.astype(np.int64), scores.astype(np.float32)

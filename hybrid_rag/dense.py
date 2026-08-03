"""Dense retrieval over an in-memory vector index."""

from __future__ import annotations

import numpy as np

from .corpus import Passage
from .embeddings import Embedder, default_embedder


class DenseRetriever:
    """Cosine-similarity search. Vectors are L2-normalised, so the dot product
    is the cosine and the whole index is one matrix multiply."""

    name = "dense"

    def __init__(self, passages: list[Passage], embedder: Embedder | None = None):
        if not passages:
            raise ValueError("DenseRetriever requires at least one passage")
        self.passages = passages
        self.ids = [p.id for p in passages]
        self.embedder = embedder or default_embedder()
        self._matrix = self.embedder.fit_transform([p.searchable for p in passages])

    def score(self, query: str) -> dict[str, float]:
        vector = self.embedder.transform([query])[0]
        sims = self._matrix @ vector
        return {pid: float(s) for pid, s in zip(self.ids, sims) if s > 0}

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.score(query)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

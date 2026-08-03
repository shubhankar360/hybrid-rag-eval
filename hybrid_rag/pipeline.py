"""Assembling retrievers into the configurations under comparison."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from .bm25 import BM25Retriever
from .corpus import Passage
from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .rerank import default_reranker


@dataclass
class RetrievalPipeline:
    """One named retrieval configuration, so the benchmark can treat every
    stack — sparse, dense, fused, reranked — through a single interface."""

    name: str
    description: str
    _search: object

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return self._search(query, k)  # type: ignore[operator]

    def timed_search(self, query: str, k: int = 10) -> tuple[list[tuple[str, float]], float]:
        started = time.perf_counter()
        results = self.search(query, k)
        return results, (time.perf_counter() - started) * 1000.0


def build_pipelines(
    passages: Sequence[Passage],
    *,
    candidate_pool: int = 20,
    rrf_k: int = 60,
) -> list[RetrievalPipeline]:
    """The four stacks under test, each adding exactly one idea to the last."""
    passages = list(passages)
    bm25 = BM25Retriever(passages)
    dense = DenseRetriever(passages)
    reranker = default_reranker(passages)

    def bm25_only(query: str, k: int):
        return bm25.search(query, k)

    def dense_only(query: str, k: int):
        return dense.search(query, k)

    def hybrid(query: str, k: int):
        return reciprocal_rank_fusion(
            [bm25.search(query, candidate_pool), dense.search(query, candidate_pool)],
            k=rrf_k,
        )[:k]

    def hybrid_reranked(query: str, k: int):
        fused = reciprocal_rank_fusion(
            [bm25.search(query, candidate_pool), dense.search(query, candidate_pool)],
            k=rrf_k,
        )[:candidate_pool]
        return reranker.rerank(query, fused, top_k=k)

    return [
        RetrievalPipeline("bm25", "Okapi BM25 only (sparse lexical baseline)", bm25_only),
        RetrievalPipeline("dense", f"Dense only ({dense.embedder.name} embeddings, cosine)", dense_only),
        RetrievalPipeline("hybrid_rrf", f"BM25 + dense fused by RRF (k={rrf_k})", hybrid),
        RetrievalPipeline("hybrid_rerank", f"Hybrid RRF then {reranker.name} over top {candidate_pool}", hybrid_reranked),
    ]

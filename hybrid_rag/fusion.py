"""Combining ranked lists from retrievers that score on different scales."""

from __future__ import annotations

from collections.abc import Sequence


def _to_ranking(results: Sequence[tuple[str, float]]) -> list[str]:
    return [doc_id for doc_id, _ in results]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[str, float]]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score each document as sum(w / (k + rank)).

    Uses rank position rather than raw score, which is what makes it safe to
    fuse BM25 (unbounded, corpus-dependent) with cosine similarity (0..1)
    without a normalisation step that would need retuning per corpus.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must align with rankings")

    fused: dict[str, float] = {}
    for weight, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(_to_ranking(ranking), start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def weighted_score_fusion(
    score_sets: Sequence[dict[str, float]],
    *,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Min-max normalise each retriever's scores, then take a weighted sum.

    Keeps score magnitude (a document that wins decisively stays ahead), at the
    cost of being sensitive to outliers within a single query's score set.
    """
    if weights is None:
        weights = [1.0] * len(score_sets)
    if len(weights) != len(score_sets):
        raise ValueError("weights must align with score_sets")

    fused: dict[str, float] = {}
    for weight, scores in zip(weights, score_sets):
        if not scores:
            continue
        values = list(scores.values())
        low, high = min(values), max(values)
        spread = high - low
        for doc_id, score in scores.items():
            norm = 1.0 if spread == 0 else (score - low) / spread
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * norm
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

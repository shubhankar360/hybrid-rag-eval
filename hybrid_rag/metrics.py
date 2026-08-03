"""Ranking metrics.

Recall@k answers "is the answer in the context we send the LLM at all", which
is the ceiling on any RAG answer. nDCG and MRR answer "is it near the top",
which drives whether it survives the context window and the model's attention.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def _ids(results: Sequence[tuple[str, float]] | Sequence[str]) -> list[str]:
    if results and isinstance(results[0], tuple):
        return [r[0] for r in results]  # type: ignore[index]
    return list(results)  # type: ignore[arg-type]


def recall_at_k(results, relevant: Collection[str], k: int) -> float:
    """Fraction of relevant passages that appear in the top k."""
    if not relevant:
        return 0.0
    top = set(_ids(results)[:k])
    return len(top & set(relevant)) / len(relevant)


def precision_at_k(results, relevant: Collection[str], k: int) -> float:
    """Fraction of the top k that is relevant."""
    if k <= 0:
        return 0.0
    top = _ids(results)[:k]
    if not top:
        return 0.0
    return sum(1 for doc_id in top if doc_id in relevant) / k


def mrr(results, relevant: Collection[str], k: int = 10) -> float:
    """Reciprocal rank of the first relevant hit; 0 if none in the top k."""
    for rank, doc_id in enumerate(_ids(results)[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results, relevant: Collection[str], k: int = 10) -> float:
    """Binary-gain nDCG@k, normalised by the best achievable ordering."""
    if not relevant:
        return 0.0
    gains = [1.0 if doc_id in relevant else 0.0 for doc_id in _ids(results)[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0

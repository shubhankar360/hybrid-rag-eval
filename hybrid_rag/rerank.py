"""Reranking: reorder a shortlist with a slower, more accurate scorer.

Reranking only ever reorders what retrieval already returned, so it can lift
precision at small k but can never recover a passage the first stage missed.
The benchmark reports both, which is how that ceiling stays visible.
"""

from __future__ import annotations

from collections.abc import Sequence

from .corpus import Passage, tokenize


class LexicalOverlapReranker:
    """Zero-dependency default: IDF-free overlap between query and passage.

    Deliberately simple and honest about it — it scores coverage of query terms
    against passage length, which is a different signal from BM25's saturated
    term frequency, but it is not a semantic model.
    """

    name = "overlap-rerank"

    def __init__(self, passages: Sequence[Passage]):
        self._by_id = {p.id: p for p in passages}

    def rerank(self, query: str, candidates: Sequence[tuple[str, float]], top_k: int | None = None):
        q_terms = set(tokenize(query))
        if not q_terms:
            return list(candidates[:top_k] if top_k else candidates)

        scored: list[tuple[str, float]] = []
        for doc_id, _ in candidates:
            passage = self._by_id.get(doc_id)
            if passage is None:
                continue
            p_terms = tokenize(passage.searchable)
            if not p_terms:
                continue
            unique = set(p_terms)
            coverage = len(q_terms & unique) / len(q_terms)
            density = len(q_terms & unique) / (len(unique) ** 0.5)
            scored.append((doc_id, coverage * 2.0 + density))

        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_k] if top_k else scored


class CrossEncoderReranker:
    """Cross-encoder reranking, used when sentence-transformers is installed.

    A cross-encoder reads query and passage together instead of embedding them
    separately, which is why it is more accurate and too slow to run over a
    whole corpus — hence a shortlist.
    """

    def __init__(self, passages: Sequence[Passage], model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.name = f"ce:{model_name}"
        self._model = CrossEncoder(model_name)
        self._by_id = {p.id: p for p in passages}

    def rerank(self, query: str, candidates: Sequence[tuple[str, float]], top_k: int | None = None):
        ids = [doc_id for doc_id, _ in candidates if doc_id in self._by_id]
        if not ids:
            return []
        pairs = [(query, self._by_id[doc_id].searchable) for doc_id in ids]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(ids, (float(s) for s in scores)), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k] if top_k else ranked


def default_reranker(passages: Sequence[Passage]):
    try:
        return CrossEncoderReranker(passages)
    except Exception:
        return LexicalOverlapReranker(passages)

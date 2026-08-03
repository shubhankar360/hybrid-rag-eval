"""Okapi BM25, implemented directly rather than pulled from a library.

Written out because the ranking function is the thing being evaluated: if it is
opaque, a bad hybrid result cannot be attributed to fusion versus scoring.
"""

from __future__ import annotations

import math
from collections import Counter

from .corpus import Passage, tokenize


class BM25Retriever:
    """Okapi BM25 with the standard k1/b parameterisation.

    k1 controls term-frequency saturation, b controls length normalisation.
    """

    name = "bm25"

    def __init__(self, passages: list[Passage], *, k1: float = 1.5, b: float = 0.75):
        if not passages:
            raise ValueError("BM25Retriever requires at least one passage")
        self.k1 = k1
        self.b = b
        self.passages = passages
        self.ids = [p.id for p in passages]

        self._docs = [Counter(tokenize(p.searchable)) for p in passages]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = sum(self._lengths) / len(self._lengths)

        doc_freq: Counter[str] = Counter()
        for doc in self._docs:
            doc_freq.update(doc.keys())

        n = len(passages)
        # Robertson/Sparck-Jones IDF with the +0.5 smoothing that keeps rare
        # terms from producing negative weights on tiny corpora.
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def score(self, query: str) -> dict[str, float]:
        terms = tokenize(query)
        scores: dict[str, float] = {}
        for i, doc in enumerate(self._docs):
            length = self._lengths[i]
            total = 0.0
            for term in terms:
                tf = doc.get(term)
                if not tf:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                total += idf * (tf * (self.k1 + 1)) / denom
            if total:
                scores[self.ids[i]] = total
        return scores

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.score(query)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

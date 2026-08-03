"""Embedding backends.

Two implementations behind one interface. The default has no network and no
model download, so `pytest` and `run_benchmark.py` work on a clean checkout;
the sentence-transformers backend is a one-line swap when a real encoder is
wanted. Benchmark numbers in the README state which backend produced them.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def fit_transform(self, texts: list[str]) -> np.ndarray: ...
    def transform(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class LsaEmbedder:
    """Offline dense vectors: word + character TF-IDF reduced by truncated SVD.

    Latent semantic indexing, not a neural encoder. It generalises past exact
    token overlap — character n-grams absorb morphology ("cancel"/"cancelling")
    and SVD folds co-occurring vocabulary together — but it cannot resolve a
    synonym that never co-occurs in the corpus. That ceiling is the point: it
    makes the dense leg genuinely different from BM25 without pretending to be
    a sentence transformer.
    """

    name = "lsa"

    def __init__(self, n_components: int = 128, random_state: int = 0):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self._vectorizer = FeatureUnion(
            [
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
            ]
        )
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._n_components = n_components
        self._fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        sparse = self._vectorizer.fit_transform(texts)
        # SVD cannot ask for more components than the matrix has rank.
        self._svd.n_components = min(self._n_components, min(sparse.shape) - 1)
        dense = self._svd.fit_transform(sparse)
        self._fitted = True
        return _l2_normalize(dense)

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit_transform on the corpus before transform")
        return _l2_normalize(self._svd.transform(self._vectorizer.transform(texts)))


class SentenceTransformerEmbedder:
    """Real neural embeddings, used when sentence-transformers is installed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.name = f"st:{model_name}"
        self._model = SentenceTransformer(model_name)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def default_embedder() -> Embedder:
    """Prefer a real encoder; fall back so a clean checkout still runs."""
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return LsaEmbedder()

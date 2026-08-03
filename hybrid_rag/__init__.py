"""Hybrid retrieval for RAG: BM25 + dense + fusion + reranking, with evaluation."""

from .corpus import Passage, Query, load_corpus, load_queries
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion, weighted_score_fusion
from .rerank import LexicalOverlapReranker, CrossEncoderReranker
from .metrics import recall_at_k, precision_at_k, mrr, ndcg_at_k
from .pipeline import RetrievalPipeline, build_pipelines

__all__ = [
    "Passage", "Query", "load_corpus", "load_queries",
    "BM25Retriever", "DenseRetriever",
    "reciprocal_rank_fusion", "weighted_score_fusion",
    "LexicalOverlapReranker", "CrossEncoderReranker",
    "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k",
    "RetrievalPipeline", "build_pipelines",
]

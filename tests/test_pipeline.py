"""End-to-end checks. These run offline: no API key, no model download."""

import pytest

from hybrid_rag.benchmark import evaluate, run, to_markdown
from hybrid_rag.corpus import load_corpus, load_queries
from hybrid_rag.pipeline import build_pipelines


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def queries():
    return load_queries()


@pytest.fixture(scope="module")
def pipelines(corpus):
    return build_pipelines(corpus)


def test_dataset_labels_point_at_real_passages(corpus, queries):
    ids = {p.id for p in corpus}
    for query in queries:
        assert query.relevant, f"{query.id} has no gold label"
        assert query.relevant <= ids, f"{query.id} references a missing passage"


def test_every_pipeline_returns_k_results(pipelines):
    for pipeline in pipelines:
        assert len(pipeline.search("how do I rotate an API key", k=5)) == 5


def test_results_are_sorted_by_descending_score(pipelines):
    for pipeline in pipelines:
        scores = [s for _, s in pipeline.search("data retention policy", k=10)]
        assert scores == sorted(scores, reverse=True)


def test_retrieval_beats_random_by_a_wide_margin(pipelines, queries):
    """A 56-passage corpus gives ~5% recall@3 by chance. Anything near that
    means the index is broken, not merely underperforming."""
    for pipeline in pipelines:
        result = evaluate(pipeline, queries)
        assert result.recall_at_3 > 0.5, f"{pipeline.name} recall@3={result.recall_at_3:.3f}"


def test_benchmark_table_lists_every_pipeline(corpus, queries):
    results = run(corpus, queries)
    table = to_markdown(results, n_passages=len(corpus), n_queries=len(queries))
    for result in results:
        assert f"`{result.pipeline}`" in table


def test_latency_is_measured(pipelines):
    _, elapsed_ms = pipelines[0].timed_search("free trial", k=5)
    assert elapsed_ms >= 0

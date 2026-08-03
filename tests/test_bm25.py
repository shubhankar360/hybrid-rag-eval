from hybrid_rag.bm25 import BM25Retriever
from hybrid_rag.corpus import Passage, load_corpus, tokenize

DOCS = [
    Passage("d1", "Rate limits", "The ingest endpoint accepts sixty requests per second per API key."),
    Passage("d2", "Refunds", "Refunds are issued for the current billing period only."),
    Passage("d3", "Trials", "New workspaces start with a fourteen day trial of the Growth plan."),
]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("The API key, is revoked!") == ["api", "key", "revoked"]


def test_tokenize_can_keep_stopwords():
    assert "the" in tokenize("The API key", drop_stopwords=False)


def test_exact_term_ranks_first():
    bm25 = BM25Retriever(DOCS)
    assert bm25.search("refunds billing period", k=1)[0][0] == "d2"


def test_unmatched_query_returns_nothing():
    assert BM25Retriever(DOCS).search("zzzz nonexistent term") == []


def test_scores_are_non_negative_across_the_real_corpus():
    corpus = load_corpus()
    bm25 = BM25Retriever(corpus)
    for query in ["api key rotation", "data retention", "single sign on"]:
        assert all(score >= 0 for _, score in bm25.search(query, k=10))


def test_k_truncates_results():
    bm25 = BM25Retriever(load_corpus())
    assert len(bm25.search("data", k=3)) <= 3


def test_empty_corpus_is_rejected():
    try:
        BM25Retriever([])
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty corpus")

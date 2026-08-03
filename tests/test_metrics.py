import math

from hybrid_rag.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

RANKING = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]


def test_recall_counts_only_within_k():
    assert recall_at_k(RANKING, {"c"}, 2) == 0.0
    assert recall_at_k(RANKING, {"c"}, 3) == 1.0


def test_recall_is_fraction_of_relevant_found():
    assert recall_at_k(RANKING, {"a", "d"}, 2) == 0.5


def test_precision_is_fraction_of_k_that_is_relevant():
    assert precision_at_k(RANKING, {"a", "b"}, 4) == 0.5


def test_mrr_uses_first_hit():
    assert mrr(RANKING, {"b", "d"}) == 0.5
    assert mrr(RANKING, {"zzz"}) == 0.0


def test_ndcg_is_one_for_a_perfect_ordering():
    assert math.isclose(ndcg_at_k(RANKING, {"a", "b"}, 4), 1.0)


def test_ndcg_penalises_a_late_hit():
    early = ndcg_at_k(RANKING, {"a"}, 4)
    late = ndcg_at_k(RANKING, {"d"}, 4)
    assert early > late


def test_metrics_accept_bare_ids():
    assert recall_at_k(["a", "b"], {"b"}, 2) == 1.0


def test_no_relevant_documents_scores_zero():
    assert recall_at_k(RANKING, set(), 3) == 0.0
    assert ndcg_at_k(RANKING, set(), 3) == 0.0

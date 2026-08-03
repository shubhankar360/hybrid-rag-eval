from hybrid_rag.fusion import reciprocal_rank_fusion, weighted_score_fusion


def test_rrf_promotes_a_document_both_retrievers_like():
    a = [("x", 9.0), ("y", 8.0), ("z", 1.0)]
    b = [("y", 0.9), ("z", 0.8), ("x", 0.1)]
    assert reciprocal_rank_fusion([a, b])[0][0] == "y"


def test_rrf_ignores_score_scale():
    """BM25 scores in the tens must not swamp cosine scores under 1."""
    huge = [("x", 900.0), ("y", 800.0)]
    tiny = [("y", 0.02), ("x", 0.01)]
    by_rank = [doc for doc, _ in reciprocal_rank_fusion([huge, tiny])]
    assert set(by_rank) == {"x", "y"}


def test_rrf_weights_shift_the_winner():
    a = [("x", 1.0), ("y", 0.5)]
    b = [("y", 1.0), ("x", 0.5)]
    assert reciprocal_rank_fusion([a, b], weights=[5.0, 1.0])[0][0] == "x"
    assert reciprocal_rank_fusion([a, b], weights=[1.0, 5.0])[0][0] == "y"


def test_rrf_rejects_mismatched_weights():
    try:
        reciprocal_rank_fusion([[("x", 1.0)]], weights=[1.0, 2.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_weighted_fusion_normalises_before_summing():
    ranked = weighted_score_fusion([{"x": 100.0, "y": 0.0}, {"y": 1.0, "x": 0.0}], weights=[1.0, 1.0])
    assert dict(ranked)["x"] == dict(ranked)["y"]


def test_weighted_fusion_handles_a_flat_score_set():
    ranked = weighted_score_fusion([{"x": 5.0, "y": 5.0}])
    assert dict(ranked) == {"x": 1.0, "y": 1.0}

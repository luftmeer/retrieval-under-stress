from collections import Counter
from types import SimpleNamespace

import pytest

from retrieval_lab.diagnostics import measure, select_cases
from retrieval_lab.retrieval import live


def test_cases_are_distinct_and_nonpositive_examples_are_labelled():
    scores = {f"q{i}": {"ndcg": 1.0} for i in range(8)}
    matrix = {
        condition: {method: scores for method in ("bm25", "dense", "hybrid")}
        for condition in ("clean", "typo_light")
    }

    cases = select_cases(matrix)

    assert len(cases) == 6
    assert len({row["query_id"] for row in cases}) == 6
    assert cases[0]["query_id"] == "q0"
    assert all(not row["positive_example"] for row in cases)


def test_timing_warmups_rotation_and_ranking_validation():
    calls = Counter()

    def search(method, text):
        calls[method] += 1
        return ["d1"]

    ticks = iter(range(0, 100_000_000, 1_000_000))
    queries = [("clean", "q1", "claim")]
    expected = {
        "clean": {method: {"q1": ["d1"]} for method in ("bm25", "dense", "hybrid")}
    }

    rows = measure(
        search,
        queries,
        expected,
        clock=lambda: next(ticks),
    )

    assert len(rows) == 9
    assert set(calls.values()) == {13}
    assert all(row["ms"] == 1.0 for row in rows)

    first_methods = [row["method"] for row in rows if row["method_position"] == 1]
    assert first_methods == ["bm25", "dense", "hybrid"]

    expected["clean"]["dense"]["q1"] = ["different"]

    with pytest.raises(ValueError, match="Ranking changed"):
        measure(search, queries, expected)


def test_hybrid_runs_both_components_on_every_request(monkeypatch):
    calls = []

    def lexical_search(*args, **kwargs):
        calls.append("bm25")
        return [SimpleNamespace(document_id="b")]

    def encode(*args, **kwargs):
        calls.append("encode")
        return [[1.0]]

    def dense_search(*args, **kwargs):
        calls.append("dense")
        return [{"document_id": "d"}]

    def fuse(lexical, dense, **kwargs):
        assert lexical == ["b"]
        assert dense == ["d"]
        calls.append("fusion")
        return [{"document_id": "b"}]

    monkeypatch.setattr(live, "reciprocal_rank_fusion", fuse)

    session = live.SearchSession(
        SimpleNamespace(search=lexical_search),
        SimpleNamespace(encode=encode),
        SimpleNamespace(search=dense_search),
    )

    session.search("hybrid", "claim")
    session.search("hybrid", "claim")

    assert calls == [
        "bm25",
        "encode",
        "dense",
        "fusion",
        "bm25",
        "encode",
        "dense",
        "fusion",
    ]

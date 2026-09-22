import pytest

from retrieval_lab.metrics import score_query


def test_hand_calculated_ranking() -> None:
    scores = score_query(
        ["d3", "d1", "d2"],
        {"d1": 1, "d2": 1},
        k=3,
    )

    assert scores["recall"] == pytest.approx(1.0)
    assert scores["rr"] == pytest.approx(0.5)
    assert scores["ndcg"] == pytest.approx(0.6934264036172708)


def test_perfect_ranking() -> None:
    scores = score_query(
        ["d1", "d2", "d3"],
        {"d1": 1, "d2": 1},
        k=3,
    )

    assert scores == {
        "recall": 1.0,
        "rr": 1.0,
        "ndcg": 1.0,
    }


def test_recall_uses_all_relevant_documents() -> None:
    scores = score_query(
        ["d1", "d3"],
        {"d1": 1, "d2": 1},
        k=2,
    )

    assert scores["recall"] == pytest.approx(0.5)
    assert scores["rr"] == pytest.approx(1.0)
    assert scores["ndcg"] == pytest.approx(0.6131471927654584)


@pytest.mark.parametrize(
    "ranking",
    [
        [],
        ["unknown"],
        ["d2", "d3", "d1"],
    ],
)
def test_no_relevant_result_within_cutoff(
    ranking: list[str],
) -> None:
    scores = score_query(ranking, {"d1": 1}, k=2)

    assert scores == {
        "recall": 0.0,
        "rr": 0.0,
        "ndcg": 0.0,
    }


def test_ndcg_uses_linear_graded_relevance() -> None:
    scores = score_query(
        ["low", "high"],
        {"high": 3, "low": 1},
        k=2,
    )

    assert scores["recall"] == 1.0
    assert scores["rr"] == 1.0
    assert scores["ndcg"] == pytest.approx(0.7967075809905066)


def test_duplicate_documents_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        score_query(["d1", "d1"], {"d1": 1}, k=2)


def test_query_without_positive_judgments_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive judgment"):
        score_query(["d1"], {"d1": 0}, k=1)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_cutoff_is_rejected(k: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        score_query(["d1"], {"d1": 1}, k=k)

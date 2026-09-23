import pytest

from retrieval_lab.retrieval.rrf import reciprocal_rank_fusion


def test_known_rankings_produce_expected_scores() -> None:
    hits = reciprocal_rank_fusion(
        ["d01", "d02", "d03"],
        ["d02", "d04", "d01"],
        top_k=4,
    )

    assert [hit["document_id"] for hit in hits] == [
        "d02",
        "d01",
        "d04",
        "d03",
    ]
    assert [hit["rank"] for hit in hits] == [1, 2, 3, 4]
    assert [hit["score"] for hit in hits] == pytest.approx(
        [
            1 / 62 + 1 / 61,
            1 / 61 + 1 / 63,
            1 / 62,
            1 / 63,
        ]
    )


def test_fusion_happens_before_output_truncation() -> None:
    hits = reciprocal_rank_fusion(
        ["d01", "d02"],
        ["d03", "d02"],
        top_k=1,
    )

    assert hits[0]["document_id"] == "d02"


def test_ties_use_document_id_and_sources_have_equal_weight() -> None:
    left = ["d02", "d01"]
    right = ["d01", "d02"]

    hits = reciprocal_rank_fusion(left, right, top_k=2)

    assert [hit["document_id"] for hit in hits] == ["d01", "d02"]
    assert hits == reciprocal_rank_fusion(right, left, top_k=2)


def test_one_empty_source_preserves_other_source_order() -> None:
    hits = reciprocal_rank_fusion(
        [],
        ["d03", "d01"],
        top_k=10,
    )

    assert [hit["document_id"] for hit in hits] == ["d03", "d01"]
    assert [hit["score"] for hit in hits] == pytest.approx([1 / 61, 1 / 62])


def test_two_empty_sources_return_no_hits() -> None:
    assert reciprocal_rank_fusion([], [], top_k=10) == []


def test_duplicate_document_within_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        reciprocal_rank_fusion(
            ["d01", "d01"],
            [],
            top_k=10,
        )


@pytest.mark.parametrize(
    ("top_k", "rrf_k", "message"),
    [
        (0, 60, "top_k"),
        (True, 60, "top_k"),
        (10, -1, "rrf_k"),
        (10, True, "rrf_k"),
    ],
)
def test_invalid_parameters_are_rejected(
    top_k: int,
    rrf_k: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reciprocal_rank_fusion(
            [],
            [],
            top_k=top_k,
            rrf_k=rrf_k,
        )

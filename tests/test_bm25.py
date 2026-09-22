from pathlib import Path

import pytest

from retrieval_lab.data.loader import Document, load_dataset
from retrieval_lab.retrieval.bm25 import BM25Retriever, tokenize

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def documents() -> dict[str, Document]:
    return {
        "d3": Document(title="", text="bread oven"),
        "d1": Document(title="", text="bicycle chain"),
        "d2": Document(title="", text="plant soil"),
    }


def test_tokenizer_policy() -> None:
    assert tokenize("Chain, café! IL-6_gene") == [
        "chain",
        "café",
        "il",
        "6",
        "gene",
    ]


def test_matching_document_rangs_first(
    documents: dict[str, Document],
) -> None:
    retriever = BM25Retriever(documents)

    hits = retriever.search("BICYCLE!", top_k=3)

    assert hits[0].document_id == "d1"
    assert hits[0].score > hits[1].score
    assert len(hits) == 3


def test_title_is_searchable() -> None:
    documents = {
        "d1": Document(title="Bicycle", text="chain"),
        "d2": Document(title="Plant", text="soil"),
        "d3": Document(title="Bread", text="oven"),
    }

    hits = BM25Retriever(documents).search("bicycle", top_k=1)

    assert hits[0].document_id == "d1"


def test_unknown_query_has_zero_scores_and_stable_ties(
    documents: dict[str, Document],
) -> None:
    hits = BM25Retriever(documents).search("zzzzunknown", top_k=3)

    assert [hit.document_id for hit in hits] == ["d1", "d2", "d3"]
    assert all(hit.score == 0.0 for hit in hits)


def test_corpus_order_does_not_change_results(documents: dict[str, Document]) -> None:
    reversed_documents = dict(reversed(list(documents.items())))

    first = BM25Retriever(documents).search("bicycle", top_k=3)
    second = BM25Retriever(reversed_documents).search("bicycle", top_k=3)

    assert first == second


def test_top_k_limits_results(
    documents: dict[str, Document],
) -> None:
    retriever = BM25Retriever(documents)

    assert len(retriever.search("bicycle", top_k=2)) == 2
    assert len(retriever.search("bicycle", top_k=100)) == 3


@pytest.mark.parametrize("query", ["", "   ", "!!!"])
def test_query_without_tokens_returns_no_hits(
    documents: dict[str, Document],
    query: str,
) -> None:
    assert BM25Retriever(documents).search(query, top_k=3) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(
    documents: dict[str, Document],
    top_k: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        BM25Retriever(documents).search("bicycle", top_k=top_k)


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        BM25Retriever({})


def test_document_without_tokens_is_rejected() -> None:
    documents = {
        "d1": Document(title="", text="!!!"),
    }

    with pytest.raises(ValueError, match="no searchable tokens"):
        BM25Retriever(documents)


def test_existing_fixture_retrieves_chain_documents() -> None:
    dataset = load_dataset(FIXTURES, FIXTURES / "qrels.tsv")
    retriever = BM25Retriever(dataset.documents)

    hits = retriever.search("bicycle chain", top_k=2)

    assert {hit.document_id for hit in hits} == {"d01", "d02"}

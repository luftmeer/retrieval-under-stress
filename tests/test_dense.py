from contextlib import closing

import numpy as np
import pytest
from qdrant_client import QdrantClient

from retrieval_lab.retrieval.dense import DenseIndex, point_id


@pytest.fixture
def client():
    with closing(QdrantClient(":memory:")) as instance:
        yield instance


def test_exact_search_returns_original_document_ids(client) -> None:
    index = DenseIndex(client, "test_documents", dimension=2)

    index.upsert(
        ["d03", "d01", "d02"],
        np.array(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )

    hits = index.search(np.array([1.0, 0.0]), top_k=3)

    assert [hit["document_id"] for hit in hits] == [
        "d01",
        "d02",
        "d03",
    ]
    assert [hit["score"] for hit in hits] == pytest.approx([1.0, 0.0, -1.0])


def test_repeated_indexing_does_not_duplicate_points(client) -> None:
    index = DenseIndex(client, "repeat", dimension=2)
    document_ids = ["d01", "d02"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])

    index.upsert(document_ids, vectors)
    index.upsert(document_ids, vectors)

    assert client.count("repeat", exact=True).count == 2

    stored = client.retrieve(
        collection_name="repeat",
        ids=[point_id("d01")],
        with_payload=True,
    )

    assert stored[0].payload["document_id"] == "d01"


def test_ties_use_document_id_within_returned_set(client) -> None:
    index = DenseIndex(client, "ties", dimension=2)
    index.upsert(
        ["d02", "d01"],
        np.array([[1.0, 0.0], [1.0, 0.0]]),
    )

    hits = index.search(np.array([1.0, 0.0]), top_k=2)

    assert [hit["document_id"] for hit in hits] == ["d01", "d02"]


def test_invalid_vector_shape_is_rejected(client) -> None:
    index = DenseIndex(client, "shape", dimension=2)

    with pytest.raises(ValueError, match="shape"):
        index.upsert(["d01"], np.array([[1.0, 0.0, 0.0]]))


def test_nonfinite_vector_is_rejected(client) -> None:
    index = DenseIndex(client, "finite", dimension=2)

    with pytest.raises(ValueError, match="finite"):
        index.upsert(["d01"], np.array([[np.nan, 1.0]]))


def test_incompatible_collection_is_rejected(client) -> None:
    DenseIndex(client, "existing", dimension=2)

    with pytest.raises(ValueError, match="configuration mismatch"):
        DenseIndex(client, "existing", dimension=3)

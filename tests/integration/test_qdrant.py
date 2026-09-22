import os
from contextlib import closing
from uuid import uuid4

import numpy as np
import pytest
from qdrant_client import QdrantClient

from retrieval_lab.retrieval.dense import DenseIndex


@pytest.mark.integration
def test_server_exact_search_and_repeat_indexing() -> None:
    url = os.environ.get("RETRIEVAL_LAB_QDRANT_URL")

    if not url:
        pytest.skip("Set RETRIEVAL_LAB_QDRANT_URL to test the server")

    collection = f"retrieval_lab_test_{uuid4().hex}"

    with closing(QdrantClient(url=url, timeout=60)) as client:
        try:
            index = DenseIndex(client, collection, dimension=2)
            ids = ["left", "up", "right"]
            vectors = np.array(
                [
                    [-1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            )

            index.upsert(ids, vectors)
            index.upsert(ids, vectors)

            hits = index.search(np.array([1.0, 0.0]), top_k=3)

            assert [hit["document_id"] for hit in hits] == [
                "right",
                "up",
                "left",
            ]
            assert client.count(collection, exact=True).count == 3
        finally:
            if client.collection_exists(collection):
                client.delete_collection(collection)

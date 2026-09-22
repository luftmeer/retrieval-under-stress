"""Exact cosine retrieval using Qdrant."""

from math import isfinite
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models


def point_id(document_id: str) -> str:
    """Map an original document ID to a stable Qdrant UUID."""
    return str(
        uuid5(
            NAMESPACE_URL,
            f"retrieval-under-stress/document/{document_id}",
        )
    )


class DenseIndex:
    """Store document vectors and return original document IDs."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dimension: int,
    ) -> None:
        if type(dimension) is not int or dimension < 1:
            raise ValueError("dimension must be a positive integer")

        self.client = client
        self.collection_name = collection_name
        self.dimension = dimension

        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                ),
            )

        info = client.get_collection(collection_name)
        parameters = info.config.params.vectors

        if (
            not isinstance(parameters, models.VectorParams)
            or parameters.size != dimension
            or parameters.distance != models.Distance.COSINE
        ):
            raise ValueError("Collection vector configuration mismatch")

    def upsert(
        self,
        document_ids: list[str],
        vectors: np.ndarray,
    ) -> None:
        """Insert or replace vectors using stable point IDs."""
        if not document_ids:
            raise ValueError("Cannot index an empty corpus")

        if any(
            not isinstance(document_id, str) or not document_id
            for document_id in document_ids
        ):
            raise ValueError("Document IDs must be nonempty strings.")

        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Duplicate document IDs")

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.shape != (len(document_ids), self.dimension):
            raise ValueError("Vector shape does not match IDs and dimension")

        if not np.isfinite(vectors).all():
            raise ValueError("Vectors must contain finite values")

        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Zero vectors are not valid for cosine search")

        for start in range(0, len(document_ids), 128):
            stop = min(start + 128, len(document_ids))

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id(document_ids[position]),
                        vector=vectors[position].tolist(),
                        payload={
                            "document_id": document_ids[position],
                        },
                    )
                    for position in range(start, stop)
                ],
                wait=True,
            )

        count = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count

        if count != len(document_ids):
            raise ValueError(
                "Collection count mismatch; unexpected points may be present"
            )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> list[dict]:
        """Return exact cosine results, sorted within the returned set."""
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        vector = np.asarray(query_vector, dtype=np.float32)

        if vector.shape != (self.dimension,):
            raise ValueError("Query vector has an invalid shape")

        if not np.isfinite(vector).all():
            raise ValueError("Query vector must contain finite values")

        if np.linalg.norm(vector) == 0:
            raise ValueError("Query vector must be non-zero")

        points = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            limit=top_k,
            search_params=models.SearchParams(exact=True),
            with_payload=True,
            with_vectors=False,
        ).points

        hits = []

        for point in points:
            document_id = (point.payload or {}).get("document_id")
            score = float(point.score)

            if not isinstance(document_id, str):
                raise ValueError("Search result has no original document ID")

            if not isfinite(score):
                raise ValueError("Search result has a non-finite score")

            hits.append(
                {
                    "document_id": document_id,
                    "score": score,
                }
            )

        return sorted(hits, key=lambda hit: (-hit["score"], hit["document_id"]))

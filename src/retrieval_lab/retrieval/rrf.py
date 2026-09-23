"""Combine two document rankings using reciprocal rank fusion."""


def reciprocal_rank_fusion(
    bm25_ids: list[str],
    dense_ids: list[str],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse one-based ranks with equal weights and zero for absence."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    if type(rrf_k) is not int or rrf_k < 0:
        raise ValueError("rrf_k must be a non-negative integer")

    scores: dict[str, float] = {}

    for ranking in (bm25_ids, dense_ids):
        seen = set()

        for rank, document_id in enumerate(ranking, start=1):
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("Document IDs must be non-empty strings")
            if document_id in seen:
                raise ValueError("Duplicate document ID within a ranking")

            seen.add(document_id)

            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (rrf_k + rank)

    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))

    return [
        {"rank": rank, "document_id": doc_id, "score": scores[doc_id]}
        for rank, doc_id in enumerate(ordered[:top_k], start=1)
    ]

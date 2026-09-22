"""Retrieval metrics calculated from ordered document IDs."""

from math import log2


def score_query(
    ranked_ids: list[str],
    judgments: dict[str, int],
    k: int,
) -> dict[str, float]:
    """Calculate recall, reciprocal rank, and linear-gain nDCG at k."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")

    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("Ranking contains duplicate document IDs")

    if any(type(grade) is not int or grade < 0 for grade in judgments.values()):
        raise ValueError("Judgments must be non-negative integers")

    relevant = {document_id for document_id, grade in judgments.items() if grade > 0}

    if not relevant:
        raise ValueError("Query has no positive judgment")

    retrieved = ranked_ids[:k]

    relevant_retrieved = sum(document_id in relevant for document_id in retrieved)
    recall = relevant_retrieved / len(relevant)

    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, document_id in enumerate(retrieved, start=1)
            if document_id in relevant
        ),
        0.0,
    )
    dcg = sum(
        judgments.get(document_id, 0) / log2(rank + 1)
        for rank, document_id in enumerate(retrieved, start=1)
    )

    ideal_grades = sorted(judgments.values(), reverse=True)[:k]
    ideal_dcg = sum(
        grade / log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1)
    )

    return {"recall": recall, "rr": reciprocal_rank, "ndcg": dcg / ideal_dcg}

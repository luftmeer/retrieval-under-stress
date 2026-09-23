"""Fuse compatible saved runs without rerunning either retriever."""

import hashlib
import json
import re
from copy import deepcopy
from math import isfinite
from pathlib import Path

from retrieval_lab.config import RunConfig
from retrieval_lab.retrieval.rrf import reciprocal_rank_fusion


def _read_source(
    path: Path,
    expected_method: str,
) -> tuple[dict, str]:
    raw = path.read_bytes()
    run = json.loads(raw)

    try:
        if not isinstance(run, dict) or run["schema_version"] != 1:
            raise ValueError("Unsupported retrieval-run schema")

        metadata = run["metadata"]
        results = run["results"]

        if metadata["method"] != expected_method:
            raise ValueError(f"Expected a {expected_method} run")

        depth = metadata["config"]["top_k"]
        if type(depth) is not int or depth < 1:
            raise ValueError("Invalid source retrieval depth")

        documents = metadata["documents"]
        if type(documents) is not int or documents < 1:
            raise ValueError("Invalid document count")

        if not isinstance(results, dict) or not results:
            raise ValueError("Results must contain evaluation queries")
        if metadata["queries"] != len(results):
            raise ValueError("Query count does not match results")

        for name in ("corpus", "queries", "qrels"):
            digest = metadata["inputs"][name]["sha256"]
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError(f"Invalid {name} SHA-256")

        for query_id, hits in results.items():
            if not isinstance(query_id, str) or not query_id:
                raise ValueError("Invalid query ID")
            if not isinstance(hits, list) or len(hits) > depth:
                raise ValueError("Invalid ranking length")

            seen = set()

            for position, hit in enumerate(hits, start=1):
                rank = hit["rank"]
                if type(rank) is not int or rank != position:
                    raise ValueError("Invalid stored rank")

                doc_id = hit["document_id"]
                if not isinstance(doc_id, str) or not doc_id:
                    raise ValueError("Invalid document ID")
                if doc_id in seen:
                    raise ValueError("Duplicate document ID within a ranking")
                seen.add(doc_id)

                score = hit["score"]
                if type(score) not in (int, float) or not isfinite(score):
                    raise ValueError("Invalid source score")

    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed retrieval run: {path}") from exc

    return run, hashlib.sha256(raw).hexdigest()


def build_hybrid_run(
    bm25_path: Path,
    dense_path: Path,
    settings: RunConfig,
    rrf_k: int = 60,
) -> dict:
    """Validate compatibility, fuse queries, and record provenance."""
    bm25, bm25_sha = _read_source(bm25_path, "bm25")
    dense, dense_sha = _read_source(dense_path, "dense")

    left = bm25["metadata"]
    right = dense["metadata"]

    for name in ("corpus", "queries", "qrels"):
        if left["inputs"][name]["sha256"] != right["inputs"][name]["sha256"]:
            raise ValueError(f"Source {name} hash mismatch")

    if set(bm25["results"]) != set(dense["results"]):
        raise ValueError("Source query IDs differ")
    if left["documents"] != right["documents"]:
        raise ValueError("Source document counts differ")

    candidate_depth = left["config"]["top_k"]

    if candidate_depth != right["config"]["top_k"]:
        raise ValueError("Source retrieval depths differ")
    if settings.top_k > candidate_depth:
        raise ValueError("Output top_k exceeds source retrieval depth")

    results = {}

    for query_id in sorted(bm25["results"]):
        results[query_id] = reciprocal_rank_fusion(
            [hit["document_id"] for hit in bm25["results"][query_id]],
            [hit["document_id"] for hit in dense["results"][query_id]],
            top_k=settings.top_k,
            rrf_k=rrf_k,
        )

    return {
        "schema_version": 1,
        "metadata": {
            "method": "hybrid_rrf",
            "config": settings.model_dump(),
            "documents": left["documents"],
            "queries": len(results),
            "inputs": deepcopy(left["inputs"]),
            "fusion": {
                "implementation": "retrieval-lab-rrf-v1",
                "rrf_k": rrf_k,
                "rank_base": 1,
                "weights": {"bm25": 1.0, "dense": 1.0},
                "candidate_depth_per_source": candidate_depth,
                "missing_document_contribution": 0.0,
            },
            "tie_break": "score descending, document_id ascending",
            "source_runs": {
                "bm25": {
                    "path": str(bm25_path),
                    "sha256": bm25_sha,
                },
                "dense": {
                    "path": str(dense_path),
                    "sha256": dense_sha,
                },
            },
        },
        "results": results,
    }

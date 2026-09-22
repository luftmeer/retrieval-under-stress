"""Evaluate saved retrieval runs against matching dataset inputs."""

import json
from math import isfinite
from pathlib import Path
from statistics import fmean

from retrieval_lab.data.download import sha256_file
from retrieval_lab.data.loader import load_dataset
from retrieval_lab.metrics import score_query


def require_mapping(value: object, label: str) -> dict:
    """Reject malformed JSON objects with a readable error."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def evaluate_run(
    run_path: Path,
    data_dir: Path,
    qrels_path: Path,
    k: int,
) -> dict[str, object]:
    """Evaluate stored ranks without rerunning or reordering retrieval."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")

    run = require_mapping(
        json.loads(run_path.read_text(encoding="utf-8")),
        "Run",
    )

    if run.get("schema_version") != 1:
        raise ValueError("Unsupported run schema_version")

    metadata = require_mapping(run.get("metadata"), "Metadata")
    config = require_mapping(metadata.get("config"), "Config")
    inputs = require_mapping(metadata.get("inputs"), "Inputs")
    results = require_mapping(run.get("results"), "Results")

    retrieval_depth = config.get("top_k")

    if type(retrieval_depth) is not int or retrieval_depth < 1:
        raise ValueError("Run top_k must be a positive integer")

    if k > retrieval_depth:
        raise ValueError("Evaluation k exceeds the run's retrieval depth")

    input_paths = {
        "corpus": data_dir / "corpus.jsonl",
        "queries": data_dir / "queries.jsonl",
        "qrels": qrels_path,
    }

    verified_inputs = {}

    for name, path in input_paths.items():
        recorded = require_mapping(inputs.get(name), f"Input {name}")
        actual_hash = sha256_file(path)

        if recorded.get("sha256") != actual_hash:
            raise ValueError(f"Input hash mismatch: {name}")

        verified_inputs[name] = {
            "path": str(path),
            "sha256": actual_hash,
        }

    dataset = load_dataset(data_dir, qrels_path)

    if set(results) != set(dataset.queries):
        raise ValueError("Run query IDs do not match evaluation query IDs")

    per_query = {}

    for query_id in sorted(dataset.queries):
        hits = results[query_id]

        if not isinstance(hits, list):
            raise ValueError(f"{query_id}: results must be a list")

        if len(hits) > min(retrieval_depth, len(dataset.documents)):
            raise ValueError(f"{query_id}: too many retrieved documents")

        ranked_ids = []

        for expected_rank, raw_hit in enumerate(hits, start=1):
            hit = require_mapping(raw_hit, f"{query_id} hit")
            rank = hit.get("rank")
            document_id = hit.get("document_id")
            score = hit.get("score")

            if type(rank) is not int or rank != expected_rank:
                raise ValueError(f"{query_id}: invalid stored rank")

            if not isinstance(document_id, str) or document_id not in dataset.documents:
                raise ValueError(f"{query_id}: unknown document ID")

            if type(score) not in (int, float) or not isfinite(score):
                raise ValueError(f"{query_id}: invalid score")

            ranked_ids.append(document_id)

        per_query[query_id] = score_query(
            ranked_ids,
            dataset.qrels[query_id],
            k,
        )

    aggregate = {
        f"recall@{k}": fmean(row["recall"] for row in per_query.values()),
        f"mrr@{k}": fmean(row["rr"] for row in per_query.values()),
        f"ndcg@{k}": fmean(row["ndcg"] for row in per_query.values()),
    }

    return {
        "schema_version": 1,
        "metadata": {
            "evaluator": "retrieval-lab-metrics-v1",
            "run_path": str(run_path),
            "run_sha256": sha256_file(run_path),
            "inputs": verified_inputs,
            "k": k,
            "queries": len(per_query),
            "aggregation": "arithmetic mean over all evaluation queries",
            "relevance_threshold": 1,
            "ndcg_gain": "linear",
            "ndcg_discount": "log2(rank + 1)",
            "unjudged_policy": "zero gain; retain ranking position",
            "ranking_policy": "preserve stored list order",
            "short_ranking_policy": "unreturned documents contribute zero",
        },
        "aggregate": aggregate,
        "per_query": per_query,
    }

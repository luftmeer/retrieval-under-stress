"""Build and save reproducible retrieval runs."""

import json
from importlib.metadata import version
from pathlib import Path

from retrieval_lab.config import RunConfig
from retrieval_lab.data.download import sha256_file
from retrieval_lab.data.loader import load_dataset
from retrieval_lab.retrieval.bm25 import (
    BM25_PARAMETERS,
    TOKEN_PATTERN,
    TOKENIZER_NAME,
    BM25Retriever,
)


def build_bm25_run(
    data_dir: Path,
    qrels_path: Path,
    settings: RunConfig,
) -> dict[str, object]:
    """Rank every selected query against the full document corpus."""
    dataset = load_dataset(data_dir, qrels_path)
    retriever = BM25Retriever(dataset.documents)

    results = {}

    for query_id in sorted(dataset.queries):
        hits = retriever.search(
            dataset.queries[query_id],
            top_k=settings.top_k,
        )

        results[query_id] = [
            {
                "rank": rank,
                "document_id": hit.document_id,
                "score": hit.score,
            }
            for rank, hit in enumerate(hits, start=1)
        ]

    input_paths = {
        "corpus": data_dir / "corpus.jsonl",
        "queries": data_dir / "queries.jsonl",
        "qrels": qrels_path,
    }

    return {
        "schema_version": 1,
        "metadata": {
            "method": "bm25",
            "implementation": "rank-bm25.BM25Okapi",
            "implementation_version": version("rank-bm25"),
            "parameters": dict(BM25_PARAMETERS),
            "config": settings.model_dump(),
            "tokenizer": {
                "name": TOKENIZER_NAME,
                "pattern": TOKEN_PATTERN,
                "normalization": "casefold",
                "stemming": False,
                "remove_stopwords": False,
            },
            "document_text_policy": (
                "title + newline + text; text alone when title is empty"
            ),
            "tie_break": "score descending, document_id ascending",
            "empty_query_policy": "return no hits",
            "score_filter": "none",
            "documents": len(dataset.documents),
            "queries": len(dataset.queries),
            "inputs": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for name, path in input_paths.items()
            },
        },
        "results": results,
    }


def save_run(
    run: dict[str, object],
    output: Path,
) -> None:
    """Write a JSON run without overwriting an existing result."""
    content = json.dumps(
        run, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as handle:
        handle.write(content + "\n")

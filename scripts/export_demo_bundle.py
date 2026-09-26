"""Export a compact immutable benchmark bundle for the demo UI."""

import argparse
import json
from pathlib import Path

from retrieval_lab import robustness as r
from retrieval_lab.data.loader import load_dataset
from retrieval_lab.diagnostics import validated_matrix


MAX_DEMO_RANK = 20
SNIPPET_CHARS = 280


def snippet(text: str) -> str:
    normalized = " ".join(text.split())

    if len(normalized) <= SNIPPET_CHARS:
        return normalized

    return normalized[: SNIPPET_CHARS - 1].rstrip() + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "runs",
        type=Path,
    )

    parser.add_argument(
        "output",
        type=Path,
    )

    args = parser.parse_args()

    root = Path(
        "data/processed/stress-test-v1"
    )

    frozen, matrix = validated_matrix(
        root,
        args.runs,
    )

    conditions = [
        condition
        for condition in r.CONDITIONS
        if condition in matrix
    ]

    datasets = {
        condition: load_dataset(
            root / condition,
            root / condition / "qrels.tsv",
        )
        for condition in conditions
    }

    rankings = {
        condition: {
            method: r.read_json(
                args.runs
                / f"{condition}-{method}.json"
            )["results"]
            for method in r.METHODS
        }
        for condition in conditions
    }

    clean = datasets["clean"]

    query_records = {}
    required_documents = set()

    for query_id in sorted(clean.queries):
        qrels = clean.qrels[query_id]

        record = {
            "variants": {},
            "qrels": qrels,
            "metrics": {},
            "rankings": {},
        }

        required_documents.update(
            document_id
            for document_id, grade in qrels.items()
            if grade > 0
        )

        for condition in conditions:
            dataset = datasets[condition]

            if query_id not in dataset.queries:
                continue

            record["variants"][condition] = (
                dataset.queries[query_id]
            )

            record["metrics"][condition] = {}
            record["rankings"][condition] = {}

            for method in r.METHODS:
                record["metrics"][condition][method] = (
                    matrix[condition][method][query_id]
                )

                hits = rankings[
                    condition
                ][method][query_id][:MAX_DEMO_RANK]

                compact_hits = [
                    {
                        "rank": hit["rank"],
                        "document_id": hit["document_id"],
                        "score": hit["score"],
                    }
                    for hit in hits
                ]

                record["rankings"][condition][method] = (
                    compact_hits
                )

                required_documents.update(
                    hit["document_id"]
                    for hit in compact_hits
                )

        query_records[query_id] = record

    documents = {}

    for document_id in sorted(required_documents):
        document = clean.documents[document_id]

        documents[document_id] = {
            "title": document.title,
            "snippet": snippet(document.text),
        }

    bundle = {
        "schema_version": 1,
        "title": "Retrieval Under Stress",
        "methods": list(r.METHODS),
        "conditions": conditions,
        "max_rank": MAX_DEMO_RANK,
        "rrf_k": 60,
        "candidate_depth": 100,
        "accepted_counts": frozen["accepted_counts"],
        "source": {
            "retrieval_manifest_sha256": r.sha256(
                args.runs / "manifest.json"
            ),
            "analysis_manifest_sha256": r.sha256(
                args.runs
                / "analysis"
                / "manifest.json"
            ),
            "freeze_manifest_sha256": r.sha256(
                root / "manifest.json"
            ),
        },
        "documents": documents,
        "queries": query_records,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "x",
        encoding="utf-8",
    ) as handle:
        json.dump(
            bundle,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")

    checksum = r.sha256(args.output)

    checksum_path = args.output.with_suffix(
        ".sha256"
    )

    with checksum_path.open(
        "x",
        encoding="utf-8",
    ) as handle:
        handle.write(checksum + "\n")

    print(
        f"Exported {len(query_records)} queries, "
        f"{len(documents)} documents"
    )

    print(f"Bundle: {args.output}")
    print(f"SHA-256: {checksum}")


if __name__ == "__main__":
    main()
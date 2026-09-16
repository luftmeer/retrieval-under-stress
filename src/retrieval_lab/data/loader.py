"""Load and validate datasets in BEIR's JSONL/TSV format."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    title: str
    text: str

    @property
    def search_text(self) -> str:
        """USe the same document text for every retrieval method."""
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass
class Dataset:
    documents: dict[str, Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def _read_records(
    path: Path,
    text_fields: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    """Read JSONL, checking required fields before indexing records."""
    records = {}

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            location = f"{path}:{line_number}"

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid JSON") from exc

            if not isinstance(row, dict):
                raise ValueError(f"{location}: expected a JSON object")

            for field in ("_id", *text_fields):
                if not isinstance(row.get(field), str):
                    raise ValueError(f"{location}: {field} must be a string")

            record_id = row["_id"]

            if not record_id.strip():
                raise ValueError(f"{location}: empty _id")

            if not row["text"].strip():
                raise ValueError(f"{location}: empty text")

            if record_id in records:
                raise ValueError(f"{location}: duplicate ID {record_id!r}")

            records[record_id] = {field: row[field] for field in text_fields}

    if not records:
        raise ValueError(f"{path}: no records")

    return records


def _read_qrels(
    path: Path,
    documents: dict[str, Document],
    queries: dict[str, str],
) -> dict[str, dict[str, int]]:
    """Read relevance judgments and validate their reference."""
    qrels: dict[str, dict[str, int]] = {}

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")

        if next(reader, None) != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"{path}: invalid qrels header")

        for line_number, row in enumerate(reader, start=2):
            location = f"{path}:{line_number}"

            if len(row) != 3:
                raise ValueError(f"{location}: expected three TSV columns")

            query_id, document_id, raw_score = row

            if query_id not in queries:
                raise ValueError(f"{location}: unknown query {query_id!r}")

            if document_id not in documents:
                raise ValueError(f"{location}: unknown document {document_id!r}")

            try:
                score = int(raw_score)
            except ValueError as exc:
                raise ValueError(f"{location}: score must be an integer") from exc

            if score < 0:
                raise ValueError(f"{location}: score must be non-negative")

            judgments = qrels.setdefault(query_id, {})

            if document_id in judgments:
                raise ValueError(f"{location}: duplicate judgment")

            judgments[document_id] = score

    if not qrels:
        raise ValueError(f"{path}: no judgments")

    for query_id, judgments in qrels.items():
        if not any(score > 0 for score in judgments.values()):
            raise ValueError(f"{path}: query {query_id} has no positive judgment")

    return qrels


def load_dataset(
    data_dir: Path,
    qrels_path: Path,
) -> Dataset:
    """Load the full corpus and the queries selected by a qrels file."""
    document_rows = _read_records(
        data_dir / "corpus.jsonl",
        text_fields=("title", "text"),
    )

    query_rows = _read_records(
        data_dir / "queries.jsonl",
        text_fields=("text",),
    )

    documents = {
        document_id: Document(**row) for document_id, row in document_rows.items()
    }

    all_queries = {query_id: row["text"] for query_id, row in query_rows.items()}

    qrels = _read_qrels(qrels_path, documents, all_queries)

    selected_queries = {query_id: all_queries[query_id] for query_id in qrels}

    return Dataset(
        documents=documents,
        queries=selected_queries,
        qrels=qrels,
    )

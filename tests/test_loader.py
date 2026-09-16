import json
from pathlib import Path

import pytest

from retrieval_lab.data.loader import load_dataset

FIXTURES = Path(__file__).parent / "fixtures"

DOCUMENT = '{"_id":"d1","title":"Title","text":"Body"}\n'
QUERY = '{"_id":"q1","text":"Question"}\n'
HEADER = "query-id\tcorpus-id\tscore\n"
JUDGMENTS = HEADER + "q1\td1\t1\n"


def write_dataset(
    root: Path,
    corpus: str = DOCUMENT,
    queries: str = QUERY,
    qrels: str = JUDGMENTS,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "corpus.jsonl").write_text(corpus, encoding="utf-8")
    (root / "queries.jsonl").write_text(queries, encoding="utf-8")

    qrels_path = root / "qrels.tsv"
    qrels_path.write_text(qrels, encoding="utf-8")
    return qrels_path


def test_loads_existing_fixture() -> None:
    dataset = load_dataset(FIXTURES, FIXTURES / "qrels.tsv")

    assert len(dataset.documents) == 20
    assert len(dataset.queries) == 5
    assert sum(len(rows) for rows in dataset.qrels.values()) == 8
    assert dataset.qrels["q01"] == {"d01": 1, "d02": 1}


def test_selects_queries_but_keeps_full_corpus(tmp_path: Path) -> None:
    corpus = DOCUMENT + '{"_id":"d2","title":"","text":"Distractor"}\n'
    queries = QUERY + '{"_id":"q2","text":"Another split"}\n'
    qrels_path = write_dataset(tmp_path, corpus=corpus, queries=queries)

    dataset = load_dataset(tmp_path, qrels_path)

    assert set(dataset.documents) == {"d1", "d2"}
    assert set(dataset.queries) == {"q1"}
    assert dataset.documents["d1"].search_text == "Title\nBody"
    assert dataset.documents["d2"].search_text == "Distractor"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("_id", 123),
        ("_id", " "),
        ("title", None),
        ("text", ["not", "a", "string"]),
        ("text", " "),
    ],
)
def test_rejects_invalid_document_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    document = json.loads(DOCUMENT)
    document[field] = value
    qrels_path = write_dataset(
        tmp_path,
        corpus=json.dumps(document) + "\n",
    )

    with pytest.raises(ValueError):
        load_dataset(tmp_path, qrels_path)


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("corpus.jsonl", DOCUMENT + DOCUMENT, "duplicate ID"),
        ("queries.jsonl", QUERY + QUERY, "duplicate ID"),
        ("corpus.jsonl", "{broken\n", "invalid JSON"),
        ("corpus.jsonl", "[]\n", "JSON object"),
        ("corpus.jsonl", '{"_id":"d1","title":"T"}\n', "text"),
        ("queries.jsonl", '{"_id":"q1","text":""}\n', "empty text"),
        ("corpus.jsonl", "", "no records"),
        ("qrels.tsv", HEADER + "q1\tmissing\t1\n", "unknown document"),
        ("qrels.tsv", HEADER + "missing\td1\t1\n", "unknown query"),
        ("qrels.tsv", JUDGMENTS + "q1\td1\t1\n", "duplicate judgment"),
        ("qrels.tsv", HEADER + "q1\td1\t0\n", "no positive judgment"),
        ("qrels.tsv", HEADER + "q1\td1\t-1\n", "non-negative"),
        ("qrels.tsv", HEADER + "q1\td1\t1.5\n", "integer"),
        ("qrels.tsv", HEADER + "q1\td1\n", "three TSV columns"),
        ("qrels.tsv", "wrong header\n", "header"),
        ("qrels.tsv", HEADER, "no judgments"),
    ],
)
def test_rejects_broken_dataset(
    tmp_path: Path,
    filename: str,
    content: str,
    message: str,
) -> None:
    qrels_path = write_dataset(tmp_path)
    (tmp_path / filename).write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_dataset(tmp_path, qrels_path)

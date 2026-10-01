import csv
import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_integrity() -> None:
    documents = [
        json.loads(line)
        for line in (FIXTURES / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    queries = [
        json.loads(line)
        for line in (FIXTURES / "queries.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    doc_ids = {row["_id"] for row in documents}
    query_ids = {row["_id"] for row in queries}

    assert len(documents) == len(doc_ids) == 20
    assert len(queries) == len(query_ids) == 5

    with (FIXTURES / "qrels.tsv").open(encoding="utf-8") as handle:
        qrels = list(csv.DictReader(handle, delimiter="\t"))
        print(qrels)

    assert {row["query-id"] for row in qrels} == query_ids

    for row in qrels:
        assert row["corpus-id"] in doc_ids
        assert int(row["score"]) == 1

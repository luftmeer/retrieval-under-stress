"""Smoke-test the running API against frozen benchmark rankings."""

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

METHODS = ("bm25", "dense", "hybrid")


def request_json(
    url: str,
    *,
    payload: dict | None = None,
) -> tuple[int, dict]:
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        url,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )

    try:
        with urlopen(
            request,
            timeout=120,
        ) as response:
            return response.status, json.load(response)

    except HTTPError as exc:
        return exc.code, json.load(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/stress-test-v1/clean"),
    )

    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/stress-test-v1"),
    )

    args = parser.parse_args()

    status, health = request_json(f"{args.base_url}/health")

    if status != 200:
        raise RuntimeError(f"Health check failed: {status}")

    corpus_size = sum(
        1
        for line in (args.data_dir / "corpus.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )

    if health["documents"] != corpus_size:
        raise ValueError("API corpus size does not match frozen corpus")

    if health["methods"] != list(METHODS):
        raise ValueError("API exposes unexpected retrieval methods")

    queries = {
        row["_id"]: row["text"]
        for row in map(
            json.loads,
            (args.data_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }

    runs = {
        method: json.loads(
            (args.runs / f"clean-{method}.json").read_text(encoding="utf-8")
        )
        for method in METHODS
    }

    common = set.intersection(*(set(run["results"]) for run in runs.values()))

    if not common:
        raise ValueError("Benchmark runs share no query IDs")

    query_id = sorted(common)[0]
    query = queries[query_id]

    for method in METHODS:
        expected = [hit["document_id"] for hit in runs[method]["results"][query_id][:10]]

        status, body = request_json(
            f"{args.base_url}/search",
            payload={
                "query": query,
                "method": method,
                "top_k": 10,
            },
        )

        if status != 200:
            raise RuntimeError(f"{method} search returned HTTP {status}")

        observed = [hit["document_id"] for hit in body["hits"]]

        if observed != expected:
            raise ValueError(
                f"{method}: API ranking differs from "
                f"benchmark for {query_id}\n"
                f"expected={expected}\n"
                f"observed={observed}"
            )

        ranks = [hit["rank"] for hit in body["hits"]]

        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"{method}: invalid ranks")

        if len(observed) != len(set(observed)):
            raise ValueError(f"{method}: duplicate documents")

        print(f"{method:<7} MATCH {query_id} top-{len(observed)}")

    status, _ = request_json(
        f"{args.base_url}/search",
        payload={
            "query": "test",
            "method": "hybrid",
            "top_k": 0,
        },
    )

    if status != 422:
        raise ValueError(f"Invalid request returned {status}, expected 422")

    print()
    print("Container API smoke test passed.")


if __name__ == "__main__":
    main()

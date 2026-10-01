import csv
import json
import os
import platform
import random
import subprocess
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch
from qdrant_client import QdrantClient

from retrieval_lab import robustness as r
from retrieval_lab.data.loader import load_dataset
from retrieval_lab.dense_runs import load_encoder
from retrieval_lab.diagnostics import measure, validated_matrix
from retrieval_lab.retrieval.bm25 import BM25Retriever
from retrieval_lab.retrieval.dense import DenseIndex
from retrieval_lab.retrieval.live import SearchSession

runs, output, environment_path = map(Path, sys.argv[1:4])
root = Path("data/processed/stress-test-v1")

frozen, matrix = validated_matrix(root, runs)
environment = r.read_json(environment_path)

if not all(environment.get(key) for key in ("cpu", "ram_gib", "qdrant_runtime")):
    raise ValueError("Fill in the latency environment first")

dirty = subprocess.check_output(
    ["git", "status", "--porcelain"],
    text=True,
).strip()

if dirty:
    raise ValueError("Commit the timing implementation and environment first")

queries = []
expected = {}
sample_ids = {}

for condition in r.CONDITIONS:
    if condition not in matrix:
        continue

    lines = (
        (root / condition / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    )

    texts = {row["_id"]: row["text"] for row in map(json.loads, lines)}

    ids = random.Random(f"42:{condition}").sample(
        sorted(texts),
        min(50, len(texts)),
    )

    sample_ids[condition] = ids
    queries.extend((condition, q, texts[q]) for q in ids)

    expected[condition] = {
        method: {
            q: [hit["document_id"] for hit in hits]
            for q, hits in r.read_json(runs / f"{condition}-{method}.json")[
                "results"
            ].items()
        }
        for method in r.METHODS
    }


def fingerprints():
    paths = {
        Path("pyproject.toml"),
        Path("uv.lock"),
        environment_path,
        Path("configs/dense-model.json"),
        Path("configs/retrieval.yaml"),
        *Path("src/retrieval_lab").rglob("*.py"),
        *Path("scripts").glob("*.py"),
    }

    return {str(path): r.sha256(path) for path in sorted(paths)}


def now():
    return datetime.now(timezone.utc).isoformat()


state = {
    "status": "running",
    "started_at": now(),
    "run_manifest_sha256": r.sha256(runs / "manifest.json"),
    "freeze_sha256": r.sha256(root / "manifest.json"),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip(),
    "git_dirty": False,
    "source_sha256": fingerprints(),
    "sample_ids": sample_ids,
    "seed": 42,
    "warmups_per_method": 10,
    "passes": 3,
    "candidate_depth": 100,
    "rrf_k": 60,
    "hybrid_execution": "sequential",
    "query_and_result_cache": False,
    "device": "cpu",
    "exact_search": True,
    "timing_boundary": ("query text to ordered top-100 IDs; no API/serialization"),
    "python": sys.version,
    "platform": platform.platform(),
    "hardware": environment,
    "torch_threads": torch.get_num_threads(),
    "torch_interop_threads": torch.get_num_interop_threads(),
    "thread_environment": {
        key: os.environ.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "TOKENIZERS_PARALLELISM",
        )
    },
}

output.mkdir(parents=True, exist_ok=False)


def save_state():
    (output / "manifest.json").write_text(
        json.dumps(state, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(name, rows):
    with (output / name).open(
        "x",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


save_state()

try:
    dataset = load_dataset(
        root / "clean",
        root / "clean/qrels.tsv",
    )

    encoder, revision = load_encoder(Path("configs/dense-model.json"))
    state["model"] = r.read_json(Path("configs/dense-model.json"))

    for condition in matrix:
        metadata = r.read_json(runs / f"{condition}-dense.json")["metadata"]

        if metadata["encoder"]["revision"] != revision:
            raise ValueError("Model revision differs from Day 8")

    if str(encoder.device) != "cpu":
        raise ValueError("This latency protocol requires CPU inference")

    with closing(
        QdrantClient(
            url="http://localhost:6333",
            timeout=60,
        )
    ) as client:
        state["qdrant_version"] = client.info().version

        collection = f"retrieval_lab_latency_{uuid4().hex}"
        state["collection"] = collection
        save_state()

        start = perf_counter()
        bm25 = BM25Retriever(dataset.documents)
        state["bm25_build_seconds"] = perf_counter() - start

        start = perf_counter()
        index = DenseIndex(
            client,
            collection,
            dimension=384,
        )

        document_ids = sorted(dataset.documents)
        vectors = encoder.encode(
            [
                dataset.documents[document_id].search_text
                for document_id in document_ids
            ],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        index.upsert(document_ids, vectors)

        state["dense_encode_and_index_seconds"] = perf_counter() - start

        if client.count(collection, exact=True).count != len(document_ids):
            raise ValueError("Wrong indexed document count")

        del vectors

        session = SearchSession(bm25, encoder, index)
        rows = measure(session.search, queries, expected)

    summary = []

    for condition in sample_ids:
        for method in r.METHODS:
            values = [
                row["ms"]
                for row in rows
                if row["condition"] == condition and row["method"] == method
            ]

            summary.append(
                {
                    "condition": condition,
                    "method": method,
                    "n_queries": len(sample_ids[condition]),
                    "n_requests": len(values),
                    "median_ms": r.percentile(values, 0.5),
                    "p95_ms": r.percentile(values, 0.95),
                }
            )

    validated_matrix(root, runs)

    if fingerprints() != state["source_sha256"]:
        raise ValueError("Timing inputs or code changed during execution")

    write_csv("latency.csv", rows)
    write_csv("latency-summary.csv", summary)

    state["output_sha256"] = {
        path.name: r.sha256(path) for path in output.glob("*.csv")
    }
    state["status"] = "complete"

except BaseException as exc:
    state["status"] = "failed"
    state["error"] = repr(exc)
    raise

finally:
    state["finished_at"] = now()
    save_state()

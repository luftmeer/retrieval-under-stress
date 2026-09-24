"""Frozen-data checks and paired bootstrap analysis."""

import hashlib
import json
import math
import random
from pathlib import Path
from statistics import fmean

METHODS = ("bm25", "dense", "hybrid")
METRICS = {
    "ndcg@10": "ndcg",
    "recall@10": "recall",
    "mrr@10": "rr",
}
CONDITIONS = ("clean", "typo_light", "typo_heavy", "paraphrase")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_freeze(root: Path) -> dict:
    committed = Path("data/manifests/stress-test-freeze-v1")
    expected = committed.with_suffix(".sha256").read_text().strip()

    for path in (
        committed.with_suffix(".json"),
        root / "manifest.json",
    ):
        if sha256(path) != expected:
            raise ValueError(f"Frozen manifest changed: {path}")

    manifest = read_json(root / "manifest.json")
    counts = manifest["accepted_counts"]

    if set(counts) != set(CONDITIONS) or counts["clean"] != 300:
        raise ValueError("Unexpected final conditions or clean query count")

    if any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Invalid accepted counts")

    expected_files = {
        f"{condition}/{name}"
        for condition, n in counts.items()
        if n > 0
        for name in ("corpus.jsonl", "queries.jsonl", "qrels.tsv")
    }

    if set(manifest["files"]) != expected_files:
        raise ValueError("Incomplete freeze file inventory")

    for relative, fingerprint in manifest["files"].items():
        if sha256(root / relative) != fingerprint:
            raise ValueError(f"Frozen file changed: {relative}")

    return manifest


def load_matrix(root: Path, runs: Path, frozen: dict) -> dict:
    matrix = {}

    for condition, n in frozen["accepted_counts"].items():
        if n == 0:
            continue

        lines = (root / condition / "queries.jsonl").read_text().splitlines()

        query_ids = [json.loads(line)["_id"] for line in lines]
        ids = set(query_ids)

        if len(query_ids) != n or len(ids) != n:
            raise ValueError(f"Incorrect query count: {condition}")

        matrix[condition] = {}

        for method in METHODS:
            run_path = runs / f"{condition}-{method}.json"
            run = read_json(run_path)
            report = read_json(runs / f"{condition}-{method}-evaluation.json")

            config = run["metadata"]["config"]

            if config["top_k"] != 100 or config["seed"] != 42:
                raise ValueError(f"Unexpected retrieval configuration: {run_path}")

            for name, filename in (
                ("corpus", "corpus.jsonl"),
                ("queries", "queries.jsonl"),
                ("qrels", "qrels.tsv"),
            ):
                observed = run["metadata"]["inputs"][name]["sha256"]
                expected = frozen["files"][f"{condition}/{filename}"]

                if observed != expected:
                    raise ValueError(f"Run does not match freeze: {run_path}")

            if report["metadata"]["run_sha256"] != sha256(run_path):
                raise ValueError(f"Evaluation refers to another run: {run_path}")

            if report["metadata"]["k"] != 10:
                raise ValueError("Expected evaluation cutoff 10")

            scores = report["per_query"]

            if set(scores) != ids or set(run["results"]) != ids:
                raise ValueError(f"Missing or extra query IDs: {run_path}")

            for values in scores.values():
                for key in METRICS.values():
                    value = values[key]

                    if (
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                        or not 0 <= value <= 1
                    ):
                        raise ValueError("Metrics must be finite values in [0, 1]")

            matrix[condition][method] = scores

    return matrix


def paired_values(clean: dict, stressed: dict) -> tuple:
    """Align scalar scores by parent ID, using the stressed subset."""
    ids = sorted(stressed)

    if not ids or not set(ids) <= set(clean):
        raise ValueError("Missing paired clean measurements")

    baseline = [clean[q] for q in ids]
    changed = [stressed[q] for q in ids]
    drops = [a - b for a, b in zip(baseline, changed)]

    return ids, baseline, changed, drops


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability

    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low

    return ordered[low] * (1 - weight) + ordered[high] * weight


def bootstrap_means(
    columns: dict[str, list[float]],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Each row is one parent query; all columns share every resample."""
    lengths = {len(values) for values in columns.values()}

    if len(lengths) != 1 or min(lengths) < 2:
        raise ValueError("Need aligned columns with at least two queries")

    if type(n_resamples) is not int or n_resamples < 2:
        raise ValueError("Need at least two resamples")

    if any(not math.isfinite(v) for column in columns.values() for v in column):
        raise ValueError("Non-finite bootstrap input")

    n = next(iter(lengths))
    rng = random.Random(seed)
    samples = {name: [] for name in columns}

    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]

        for name, values in columns.items():
            samples[name].append(fmean(values[i] for i in indices))

    estimates = {
        name: {
            "estimate": fmean(columns[name]),
            "ci_low": percentile(values, 0.025),
            "ci_high": percentile(values, 0.975),
        }
        for name, values in samples.items()
    }

    return estimates, samples

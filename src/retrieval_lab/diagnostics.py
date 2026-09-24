"""Saved-result validation, case selection and warm-request measurements."""

import random
from time import perf_counter_ns

from retrieval_lab import robustness as r


def validated_matrix(root, runs):
    frozen = r.verify_freeze(root)
    manifest = r.read_json(runs / "manifest.json")
    analysis = r.read_json(runs / "analysis/manifest.json")

    expected = {
        f"{condition}-{method}{suffix}.json"
        for condition, n in frozen["accepted_counts"].items()
        if n > 0
        for method in r.METHODS
        for suffix in ("", "-evaluation")
    }

    if manifest["status"] != "complete" or analysis["status"] != "complete":
        raise ValueError("Day 8 is incomplete")

    if set(manifest["output_sha256"]) != expected:
        raise ValueError("Incomplete ranking/evaluation inventory")

    freeze_hash = r.sha256(root / "manifest.json")

    if any(record["freeze_sha256"] != freeze_hash for record in (manifest, analysis)):
        raise ValueError("Different frozen benchmark")

    if analysis["run_manifest_sha256"] != r.sha256(runs / "manifest.json"):
        raise ValueError("Analysis belongs to another run")

    for directory, record in (
        (runs, manifest),
        (runs / "analysis", analysis),
    ):
        for name, fingerprint in record["output_sha256"].items():
            if r.sha256(directory / name) != fingerprint:
                raise ValueError(f"Changed result: {directory / name}")

    for condition, n in frozen["accepted_counts"].items():
        if n > 0:
            hybrid = r.read_json(runs / f"{condition}-hybrid.json")
            if hybrid["metadata"]["fusion"]["rrf_k"] != 60:
                raise ValueError("Expected RRF constant 60")

    return frozen, r.load_matrix(root, runs, frozen)


def select_cases(matrix):
    selected = []
    used = set()

    for category in ("dense_loss", "bm25_loss", "hybrid_gain"):
        candidates = []

        for condition in r.CONDITIONS[1:]:
            if condition not in matrix:
                continue

            scores = matrix[condition]

            for q in sorted(scores["bm25"]):
                if category == "hybrid_gain":
                    effect = scores["hybrid"][q]["ndcg"] - max(
                        scores[method][q]["ndcg"] for method in ("bm25", "dense")
                    )
                else:
                    method = category.removesuffix("_loss")
                    effect = (
                        matrix["clean"][method][q]["ndcg"] - scores[method][q]["ndcg"]
                    )

                candidates.append(
                    {
                        "category": category,
                        "condition": condition,
                        "query_id": q,
                        "effect": effect,
                        "positive_example": effect > 0,
                    }
                )

        candidates.sort(
            key=lambda row: (
                -row["effect"],
                row["query_id"],
                r.CONDITIONS.index(row["condition"]),
            )
        )

        taken = 0

        for row in candidates:
            if row["query_id"] in used:
                continue

            selected.append(row)
            used.add(row["query_id"])
            taken += 1

            if taken == 2:
                break

    return selected


def measure(search, queries, expected, *, clock=perf_counter_ns):
    """queries contains (condition, parent_id, text) tuples.

    search(method, text) returns ordered document IDs.
    """
    if not queries:
        raise ValueError("No timing queries")

    warmup = list(queries)
    random.Random(42).shuffle(warmup)

    for method in r.METHODS:
        for i in range(10):
            search(method, warmup[i % len(warmup)][2])

    rows = []

    for pass_index in range(3):
        ordered = list(queries)
        random.Random(42 + pass_index).shuffle(ordered)

        methods = r.METHODS[pass_index:] + r.METHODS[:pass_index]

        for condition, q, text in ordered:
            for position, method in enumerate(methods):
                start = clock()
                ids = search(method, text)
                duration = (clock() - start) / 1_000_000

                if ids[:10] != expected[condition][method][q][:10]:
                    raise ValueError(f"Ranking changed: {condition}/{method}/{q}")

                rows.append(
                    {
                        "condition": condition,
                        "query_id": q,
                        "method": method,
                        "pass": pass_index + 1,
                        "method_position": position + 1,
                        "ms": duration,
                    }
                )

    return rows

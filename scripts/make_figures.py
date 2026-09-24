"""Render saved evidence without importing retrieval code."""

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter
from importlib.metadata import distributions
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

METHODS = ("bm25", "dense", "hybrid")
CONDITIONS = ("clean", "typo_light", "typo_heavy", "paraphrase")

LABELS = {
    "clean": "Clean",
    "typo_light": "Light typos",
    "typo_heavy": "Heavy typos",
    "paraphrase": "Paraphrases (exploratory)",
}

NAMES = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
}

COLORS = {
    "bm25": "#0072B2",
    "dense": "#D55E00",
    "hybrid": "#009E73",
}

MARKERS = {
    "bm25": "o",
    "dense": "s",
    "hybrid": "^",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def checked_manifest(directory, required=()):
    manifest = json.loads((directory / "manifest.json").read_text())

    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete run: {directory}")

    inventory = manifest["output_sha256"]

    if not set(required) <= set(inventory):
        raise ValueError(f"Missing recorded outputs: {directory}")

    for name, expected in inventory.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Changed output: {directory / name}")

    return manifest


def indexed(rows, fields):
    result = {}

    for row in rows:
        key = tuple(row[field] for field in fields)

        if key in result:
            raise ValueError(f"Duplicate measurement: {key}")

        result[key] = row

    return result


def load_inputs(runs, latency):
    run = checked_manifest(runs)

    analysis = checked_manifest(
        runs / "analysis",
        ("summary.csv", "per_query.csv", "contrasts.csv"),
    )

    timing = checked_manifest(
        latency,
        ("latency.csv", "latency-summary.csv"),
    )

    for record in (analysis, timing):
        if record["run_manifest_sha256"] != sha256(runs / "manifest.json"):
            raise ValueError("Results belong to different retrieval runs")

        if record["freeze_sha256"] != run["freeze_sha256"]:
            raise ValueError("Results use different frozen benchmarks")

    conditions = [
        condition
        for condition in CONDITIONS
        if analysis["query_ids"].get(condition)
        and not (condition == "paraphrase" and len(analysis["query_ids"][condition]) < 30)
    ]

    summary = indexed(
        [
            row
            for row in read_csv(runs / "analysis/summary.csv")
            if row["metric"] == "ndcg@10"
        ],
        ("condition", "method"),
    )

    scores = indexed(
        [
            row
            for row in read_csv(runs / "analysis/per_query.csv")
            if row["metric"] == "ndcg@10"
        ],
        ("condition", "method", "query_id"),
    )

    times = indexed(
        read_csv(latency / "latency-summary.csv"),
        ("condition", "method"),
    )

    expected_counts = Counter(
        {
            (condition, method, query_id): timing["passes"]
            for condition, ids in timing["sample_ids"].items()
            for method in METHODS
            for query_id in ids
        }
    )

    actual_counts = Counter(
        (row["condition"], row["method"], row["query_id"])
        for row in read_csv(latency / "latency.csv")
    )

    if actual_counts != expected_counts:
        raise ValueError("Timing requests do not match the recorded query sample")

    rows = []

    for condition in conditions:
        ids = set(analysis["query_ids"][condition])
        sampled = timing["sample_ids"][condition]

        if not sampled or len(set(sampled)) != len(sampled):
            raise ValueError(f"Empty or duplicate timing sample: {condition}")

        if not set(sampled) <= ids:
            raise ValueError(f"Unmatched timing queries: {condition}")

        for method in METHODS:
            quality = summary[condition, method]
            latency_row = times[condition, method]

            observed = {
                query_id for c, m, query_id in scores if c == condition and m == method
            }

            if observed != ids or int(quality["n"]) != len(ids):
                raise ValueError(f"Quality query mismatch: {condition}/{method}")

            if (
                int(latency_row["n_queries"]) != len(sampled)
                or int(latency_row["n_requests"]) != len(sampled) * timing["passes"]
            ):
                raise ValueError(f"Timing count mismatch: {condition}/{method}")

            row = {
                "condition": condition,
                "method": method,
                "n_quality": len(ids),
                **{
                    key: (float(quality[key]) if quality[key] else None)
                    for key in (
                        "clean_mean",
                        "condition_mean",
                        "drop",
                        "ci_low",
                        "ci_high",
                    )
                },
                "n_latency_queries": len(sampled),
                "n_latency_requests": int(latency_row["n_requests"]),
                "p95_ms": float(latency_row["p95_ms"]),
                "timed_sample_quality": fmean(
                    float(scores[condition, method, query_id]["score"])
                    for query_id in sampled
                ),
            }

            if any(
                isinstance(value, float) and not math.isfinite(value)
                for value in row.values()
            ):
                raise ValueError(f"Non-finite plot value: {condition}/{method}")

            if row["p95_ms"] <= 0:
                raise ValueError("Logarithmic latency axis requires positive times")

            if condition != "clean":
                if row["drop"] is None:
                    raise ValueError(f"Missing degradation: {condition}/{method}")

                if len(ids) >= 2 and any(
                    row[key] is None for key in ("ci_low", "ci_high")
                ):
                    raise ValueError(
                        f"Missing degradation interval: {condition}/{method}"
                    )

            rows.append(row)

    if not rows:
        raise ValueError("No quantitative conditions to plot")

    return conditions, rows


def panels(conditions, title):
    columns = min(2, len(conditions))
    rows = math.ceil(len(conditions) / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.4 * columns, 3.8 * rows),
        squeeze=False,
        layout="constrained",
    )

    fig.suptitle(title, fontsize=14)

    flat = list(axes.flat)

    for ax in flat[len(conditions) :]:
        ax.set_visible(False)

    for ax in flat[: len(conditions)]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)

    return fig, flat[: len(conditions)]


def save(fig, output, name):
    for extension in ("png", "svg"):
        fig.savefig(
            output / f"{name}.{extension}",
            dpi=200,
        )

    plt.close(fig)


def plot_quality(conditions, data, output):
    fig, axes = panels(
        conditions,
        "Retrieval quality: full eligible query sets",
    )

    for condition, ax in zip(conditions, axes):
        for position, method in enumerate(METHODS):
            row = data[condition, method]

            ax.bar(
                position,
                row["condition_mean"],
                color=COLORS[method],
                width=0.6,
            )

            if condition != "clean":
                ax.scatter(
                    position,
                    row["clean_mean"],
                    marker="D",
                    s=55,
                    facecolors="white",
                    edgecolors="black",
                    zorder=3,
                )

        n = data[condition, "bm25"]["n_quality"]

        ax.set(
            title=f"{LABELS[condition]} | n={n}",
            ylabel="Mean nDCG@10",
            ylim=(0, 1),
        )

        ax.set_xticks(
            range(3),
            [NAMES[method] for method in METHODS],
        )

        ax.grid(axis="y", alpha=0.2)

        if condition != "clean":
            ax.legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        marker="D",
                        color="black",
                        markerfacecolor="white",
                        linestyle="none",
                        label="Clean on the same queries",
                    )
                ],
                loc="upper right",
                fontsize=8,
            )

    save(fig, output, "01-quality")


def plot_degradation(conditions, data, output):
    stressed = [condition for condition in conditions if condition != "clean"]

    if not stressed:
        raise ValueError("No quantitative stressed condition")

    bounds = [0.0]

    for condition in stressed:
        for method in METHODS:
            row = data[condition, method]

            bounds.extend(
                100 * row[key]
                for key in ("drop", "ci_low", "ci_high")
                if row[key] is not None
            )

    padding = max(
        1.0,
        (max(bounds) - min(bounds)) * 0.1,
    )

    fig, axes = panels(
        stressed,
        "Paired degradation: pointwise 95% bootstrap intervals",
    )

    for condition, ax in zip(stressed, axes):
        for position, method in enumerate(METHODS):
            row = data[condition, method]

            if row["ci_low"] is not None and row["ci_high"] is not None:
                ax.hlines(
                    position,
                    100 * row["ci_low"],
                    100 * row["ci_high"],
                    color=COLORS[method],
                    linewidth=2,
                )

            ax.scatter(
                100 * row["drop"],
                position,
                color=COLORS[method],
                marker=MARKERS[method],
                s=55,
                zorder=3,
            )

        n = data[condition, "bm25"]["n_quality"]
        suffix = " | no CI" if n < 2 else ""

        ax.set(
            title=f"{LABELS[condition]} | n={n}{suffix}",
            xlabel="nDCG@10 loss (points): clean minus stressed",
            xlim=(
                min(bounds) - padding,
                max(bounds) + padding,
            ),
            ylim=(2.5, -0.5),
        )

        ax.set_yticks(
            range(3),
            [NAMES[method] for method in METHODS],
        )

        ax.axvline(
            0,
            color="0.4",
            linestyle="--",
            linewidth=1,
        )

        ax.grid(axis="x", alpha=0.2)

    save(fig, output, "02-degradation")


def plot_tradeoff(conditions, data, output):
    values = [row["p95_ms"] for row in data.values()]

    fig, axes = panels(
        conditions,
        "Quality and latency: the same sampled queries",
    )

    for condition, ax in zip(conditions, axes):
        for method in METHODS:
            row = data[condition, method]

            ax.scatter(
                row["p95_ms"],
                row["timed_sample_quality"],
                color=COLORS[method],
                marker=MARKERS[method],
                s=75,
                label=NAMES[method],
            )

        row = data[condition, "bm25"]

        ax.set_xscale("log")

        ax.set(
            title=(
                f"{LABELS[condition]}\n"
                f"{row['n_latency_queries']} queries, "
                f"{row['n_latency_requests']} requests/method"
            ),
            xlabel="Warm p95 latency (ms; logarithmic scale)",
            ylabel="Mean nDCG@10 on timing sample",
            ylim=(0, 1),
            xlim=(
                min(values) / 1.5,
                max(values) * 1.5,
            ),
        )

        ax.grid(alpha=0.2)
        ax.legend(loc="lower left", fontsize=8)

    save(fig, output, "03-quality-latency")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("runs", type=Path)
    parser.add_argument("latency", type=Path)
    parser.add_argument("output", type=Path)

    args = parser.parse_args()

    conditions, rows = load_inputs(
        args.runs,
        args.latency,
    )

    data = indexed(
        rows,
        ("condition", "method"),
    )

    args.output.mkdir(
        parents=True,
        exist_ok=False,
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
        }
    )

    plot_quality(conditions, data, args.output)
    plot_degradation(conditions, data, args.output)
    plot_tradeoff(conditions, data, args.output)

    with (args.output / "plot-data.csv").open(
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

    sources = [
        args.runs / "manifest.json",
        args.runs / "analysis/manifest.json",
        args.latency / "manifest.json",
        Path(__file__),
    ]

    lock = Path(str(Path(__file__)) + ".lock")

    if lock.exists():
        sources.append(lock)

    manifest = {
        "status": "complete",
        "metric": "ndcg@10",
        "conditions": conditions,
        "python": platform.python_version(),
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in distributions()
        },
        "source_sha256": {str(path): sha256(path) for path in sources},
        "quality_latency_scope": ("quality computed once per sampled query"),
        "drop_scale": "100 * (clean minus stressed)",
        "output_sha256": {
            path.name: sha256(path) for path in args.output.iterdir() if path.is_file()
        },
    }

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved three figures, plotting values and manifest: {args.output}")


if __name__ == "__main__":
    main()

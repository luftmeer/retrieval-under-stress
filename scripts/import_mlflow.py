"""Import completed benchmark evidence into local MLflow."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import mlflow

METHODS = ("bm25", "dense", "hybrid")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def checked_manifest(directory, required=()):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete output: {directory}")

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
            raise ValueError(f"Duplicate row: {key}")

        result[key] = row

    return result


def numeric_metrics(row, fields):
    result = {}

    for source, target in fields.items():
        value = row[source]

        # Blank confidence limits remain missing, rather than becoming zero.
        if value is None or value == "":
            continue

        number = float(value)

        if not math.isfinite(number):
            raise ValueError(f"Non-finite metric: {source}")

        result[target] = number

    return result


def load_evidence(runs, latency, figures):
    retrieval = checked_manifest(runs)

    analysis = checked_manifest(
        runs / "analysis",
        ("summary.csv", "per_query.csv", "contrasts.csv"),
    )

    timing = checked_manifest(
        latency,
        ("latency.csv", "latency-summary.csv"),
    )

    plots = checked_manifest(
        figures,
        (
            "01-quality.png",
            "02-degradation.png",
            "03-quality-latency.png",
            "plot-data.csv",
        ),
    )

    run_hash = sha256(runs / "manifest.json")

    for record in (analysis, timing):
        if record["run_manifest_sha256"] != run_hash:
            raise ValueError("Evidence belongs to different retrieval runs")

        if record["freeze_sha256"] != retrieval["freeze_sha256"]:
            raise ValueError("Evidence uses different frozen benchmarks")

    # Day 10 recorded the source paths and hashes used for its figures.
    plot_sources = {
        Path(name).resolve(): digest for name, digest in plots["source_sha256"].items()
    }

    for path in (
        runs / "manifest.json",
        runs / "analysis/manifest.json",
        latency / "manifest.json",
    ):
        if plot_sources.get(path.resolve()) != sha256(path):
            raise ValueError(f"Figures do not match source: {path}")

    quality = indexed(
        [
            row
            for row in read_csv(runs / "analysis/summary.csv")
            if row["metric"] == "ndcg@10"
        ],
        ("condition", "method"),
    )

    times = indexed(
        read_csv(latency / "latency-summary.csv"),
        ("condition", "method"),
    )

    # Follow Day 10's quantitative inclusion decisions.
    conditions = plots["conditions"]
    children = []

    for condition in conditions:
        for method in METHODS:
            q = quality[condition, method]
            t = times[condition, method]

            metrics = numeric_metrics(
                q,
                {
                    "clean_mean": "matched_clean_ndcg10",
                    "condition_mean": "condition_ndcg10",
                    "drop": "drop_ndcg10",
                    "ci_low": "drop_ci_low",
                    "ci_high": "drop_ci_high",
                },
            )

            metrics.update(
                numeric_metrics(
                    t,
                    {
                        "median_ms": "latency_median_ms",
                        "p95_ms": "latency_p95_ms",
                    },
                )
            )

            children.append(
                {
                    "condition": condition,
                    "method": method,
                    "n_quality": int(q["n"]),
                    "n_latency_queries": int(t["n_queries"]),
                    "n_latency_requests": int(t["n_requests"]),
                    "metrics": metrics,
                }
            )

    if not children:
        raise ValueError("No quantitative results to import")

    return children


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("latency", type=Path)
    parser.add_argument("figures", type=Path)
    parser.add_argument("--name", default="stress-v1-evidence-import")
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Optional JSON file recording the imported parent run.",
    )
    args = parser.parse_args()

    # Validate and prepare all metrics before creating an MLflow run.
    children = load_evidence(
        args.runs,
        args.latency,
        args.figures,
    )

    tracking = Path(".tracking").resolve()
    tracking.mkdir(exist_ok=True)

    database = tracking / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")

    experiment_name = "Retrieval Under Stress"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=(tracking / "artifacts").as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=args.name,
    ) as parent:
        mlflow.set_tags(
            {
                "execution_kind": "saved_evidence_import",
                "retrieval_manifest_sha256": sha256(args.runs / "manifest.json"),
                "analysis_manifest_sha256": sha256(args.runs / "analysis/manifest.json"),
                "latency_manifest_sha256": sha256(args.latency / "manifest.json"),
                "figures_manifest_sha256": sha256(args.figures / "manifest.json"),
            }
        )

        mlflow.log_params(
            {
                "primary_metric": "ndcg@10",
                "drop_definition": "clean_minus_stressed",
                "drop_scale": "normalized_0_to_1",
                "imported_groups": len(children),
            }
        )

        # Copy the hash-checked outputs and manifest from each stage.
        stages = {
            "retrieval": args.runs,
            "analysis": args.runs / "analysis",
            "latency": args.latency,
            "figures": args.figures,
        }

        for stage, directory in stages.items():
            manifest = checked_manifest(directory)

            mlflow.log_artifact(
                str(directory / "manifest.json"),
                artifact_path=f"evidence/{stage}",
            )

            for name in manifest["output_sha256"]:
                relative = Path(name)

                mlflow.log_artifact(
                    str(directory / relative),
                    artifact_path=(Path("evidence") / stage / relative.parent).as_posix(),
                )

        if args.receipt is not None:
            receipt = {
                "status": "complete",
                "experiment_name": experiment_name,
                "experiment_id": experiment_id,
                "parent_run_id": parent.info.run_id,
                "tracking_uri": mlflow.get_tracking_uri(),
                "retrieval_manifest_sha256": sha256(args.runs / "manifest.json"),
                "analysis_manifest_sha256": sha256(args.runs / "analysis/manifest.json"),
                "latency_manifest_sha256": sha256(args.latency / "manifest.json"),
                "figures_manifest_sha256": sha256(args.figures / "manifest.json"),
            }

            args.receipt.parent.mkdir(parents=True, exist_ok=True)

            with args.receipt.open("x", encoding="utf-8") as handle:
                json.dump(receipt, handle, indent=2)
                handle.write("\n")

        mlflow.log_artifact(
            __file__,
            artifact_path="importer",
        )

        for filename in ("pyproject.toml", "uv.lock"):
            mlflow.log_artifact(
                str(Path("tools/tracking") / filename),
                artifact_path="importer/environment",
            )

        for row in children:
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"{row['condition']}-{row['method']}",
                nested=True,
            ):
                mlflow.log_params(
                    {key: value for key, value in row.items() if key != "metrics"}
                )

                mlflow.set_tags(
                    {
                        "quality_scope": "full_eligible_condition_set",
                        "latency_scope": "recorded_timing_sample",
                    }
                )

                mlflow.log_metrics(row["metrics"])

        print(f"Imported parent run: {parent.info.run_id}")
        print(f"Tracking database: {database}")


if __name__ == "__main__":
    main()

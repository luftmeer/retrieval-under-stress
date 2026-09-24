"""Validate final runs, then write paired estimates and bootstrap samples."""

import csv
import json
import sys
from itertools import combinations
from pathlib import Path
from statistics import fmean

from retrieval_lab import robustness as r

runs = Path(sys.argv[1])
output = runs / "analysis"
root = Path("data/processed/stress-test-v1")

frozen = r.verify_freeze(root)
run_manifest = r.read_json(runs / "manifest.json")

if run_manifest["status"] != "complete":
    raise ValueError("Retrieval matrix is incomplete")

if run_manifest["freeze_sha256"] != r.sha256(root / "manifest.json"):
    raise ValueError("Retrieval matrix used a different freeze")

expected_outputs = {
    f"{condition}-{method}{suffix}.json"
    for condition, n in frozen["accepted_counts"].items()
    if n > 0
    for method in r.METHODS
    for suffix in ("", "-evaluation")
}

if set(run_manifest["output_sha256"]) != expected_outputs:
    raise ValueError("Incomplete retrieval output inventory")

for name, fingerprint in run_manifest["output_sha256"].items():
    if r.sha256(runs / name) != fingerprint:
        raise ValueError(f"Saved result changed: {name}")

matrix = r.load_matrix(root, runs, frozen)

summary = []
contrasts = []
per_query = []
bootstrap = []
worst = []
query_sets = {}

for condition, methods in matrix.items():
    ids = sorted(methods["bm25"])
    query_sets[condition] = ids

    quantitative = condition != "paraphrase" or len(ids) >= 30
    columns = {}
    rows = []

    for method in r.METHODS:
        for metric, key in r.METRICS.items():
            clean = {q: values[key] for q, values in matrix["clean"][method].items()}
            stressed = {q: values[key] for q, values in methods[method].items()}

            paired_ids, baseline, changed, drops = r.paired_values(
                clean,
                stressed,
            )

            for q, a, b, d in zip(
                paired_ids,
                baseline,
                changed,
                drops,
            ):
                per_query.append(
                    {
                        "condition": condition,
                        "method": method,
                        "metric": metric,
                        "query_id": q,
                        "clean": a,
                        "score": b,
                        "drop": d,
                    }
                )

            if not quantitative:
                continue

            rows.append(
                {
                    "condition": condition,
                    "method": method,
                    "metric": metric,
                    "n": len(ids),
                    "clean_mean": fmean(baseline),
                    "condition_mean": fmean(changed),
                    "drop": (fmean(drops) if condition != "clean" else ""),
                    "ci_low": "",
                    "ci_high": "",
                }
            )

            columns[f"{method}|{metric}"] = drops

    if condition != "clean" and quantitative and len(ids) >= 2:
        comparisons = []

        for right, left in combinations(r.METHODS, 2):
            for metric in r.METRICS:
                name = f"{left}-minus-{right}|{metric}"

                columns[name] = [
                    a - b
                    for a, b in zip(
                        columns[f"{left}|{metric}"],
                        columns[f"{right}|{metric}"],
                    )
                ]

                comparisons.append((left, right, metric, name))

        estimates, samples = r.bootstrap_means(columns)

        for row in rows:
            name = f"{row['method']}|{row['metric']}"
            estimate = estimates[name]

            row.update(
                ci_low=estimate["ci_low"],
                ci_high=estimate["ci_high"],
            )

        for left, right, metric, name in comparisons:
            contrasts.append(
                {
                    "condition": condition,
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "n": len(ids),
                    **estimates[name],
                }
            )

        for name, values in samples.items():
            bootstrap.extend(
                {
                    "condition": condition,
                    "statistic": name,
                    "replicate": i,
                    "value": value,
                }
                for i, value in enumerate(values)
            )

    summary.extend(rows)

required = ("clean", "typo_light", "typo_heavy")
common_ids = []

if all(condition in matrix for condition in required):
    common_ids = sorted(
        set.intersection(*(set(matrix[condition]["bm25"]) for condition in required))
    )

    if common_ids:
        for method in r.METHODS:
            for metric, key in r.METRICS.items():
                means = {
                    condition: fmean(
                        matrix[condition][method][q][key] for q in common_ids
                    )
                    for condition in required
                }

                worst.append(
                    {
                        "method": method,
                        "metric": metric,
                        "n": len(common_ids),
                        **means,
                        "worst_mean": min(means.values()),
                    }
                )

output.mkdir(exist_ok=False)


def write_csv(name: str, rows: list[dict], fields: str) -> None:
    with (output / name).open(
        "x",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields.split(","),
        )
        writer.writeheader()
        writer.writerows(rows)


write_csv(
    "per_query.csv",
    per_query,
    "condition,method,metric,query_id,clean,score,drop",
)
write_csv(
    "summary.csv",
    summary,
    "condition,method,metric,n,clean_mean,condition_mean,drop,ci_low,ci_high",
)
write_csv(
    "contrasts.csv",
    contrasts,
    "condition,left,right,metric,n,estimate,ci_low,ci_high",
)
write_csv(
    "bootstrap.csv",
    bootstrap,
    "condition,statistic,replicate,value",
)
write_csv(
    "worst_condition.csv",
    worst,
    "method,metric,n,clean,typo_light,typo_heavy,worst_mean",
)

manifest = {
    "status": "complete",
    "run_manifest_sha256": r.sha256(runs / "manifest.json"),
    "freeze_sha256": r.sha256(root / "manifest.json"),
    "analysis_code_sha256": {
        "robustness": r.sha256(Path(r.__file__)),
        "script": r.sha256(Path(__file__)),
    },
    "bootstrap": {
        "method": "paired percentile",
        "resamples": 2000,
        "seed": 42,
        "confidence": 0.95,
        "quantile": "linear interpolation",
        "rng": "random.Random",
        "unit": "parent query",
        "simultaneous_intervals": False,
    },
    "query_ids": query_sets,
    "worst_condition_query_ids": common_ids,
    "qualitative_only": [
        condition
        for condition, ids in query_sets.items()
        if condition == "paraphrase" and len(ids) < 30
    ],
    "no_ci_conditions": [
        condition for condition, ids in query_sets.items() if len(ids) < 2
    ],
    "output_sha256": {path.name: r.sha256(path) for path in output.glob("*.csv")},
}

(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

print(f"Analysis saved: {output}")

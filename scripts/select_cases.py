import json
import sys
from pathlib import Path

from retrieval_lab import robustness as r
from retrieval_lab.data.loader import load_dataset
from retrieval_lab.diagnostics import select_cases, validated_matrix

runs = Path(sys.argv[1])
root = Path("data/processed/stress-test-v1")

frozen, matrix = validated_matrix(root, runs)

datasets = {
    condition: load_dataset(
        root / condition,
        root / condition / "qrels.tsv",
    )
    for condition in matrix
}

rankings = {
    condition: {
        method: r.read_json(runs / f"{condition}-{method}.json")["results"]
        for method in r.METHODS
    }
    for condition in matrix
}

cases = select_cases(matrix)

for case in cases:
    condition = case["condition"]
    q = case["query_id"]

    relevant = sorted(
        document_id
        for document_id, grade in datasets["clean"].qrels[q].items()
        if grade > 0
    )

    case["original_text"] = datasets["clean"].queries[q]
    case["modified_text"] = datasets[condition].queries[q]
    case["rankings"] = {}

    for side in ("clean", condition):
        case["rankings"][side] = {}

        for method in r.METHODS:
            hits = rankings[side][method][q]
            ranks = {hit["document_id"]: hit["rank"] for hit in hits}

            case["rankings"][side][method] = {
                "ndcg@10": matrix[side][method][q]["ndcg"],
                "relevant_ranks": {
                    document_id: ranks.get(document_id) for document_id in relevant
                },
                "top10": [
                    {
                        **hit,
                        "title": datasets["clean"].documents[hit["document_id"]].title,
                    }
                    for hit in hits[:10]
                ],
            }

    case["known_relevant_documents"] = {
        document_id: {
            "title": datasets["clean"].documents[document_id].title,
            "text": datasets["clean"].documents[document_id].text,
        }
        for document_id in relevant
    }

    case["review"] = {
        "observation": "",
        "possible_explanation": "",
        "alternative_explanation": "",
        "follow_up_test": "",
    }

output = runs / "case-selection.json"

with output.open("x", encoding="utf-8") as handle:
    json.dump(
        {
            "run_manifest_sha256": r.sha256(runs / "manifest.json"),
            "analysis_manifest_sha256": r.sha256(runs / "analysis/manifest.json"),
            "selection": (
                "dense_loss, bm25_loss, hybrid_gain; two each; distinct parents"
            ),
            "tie_break": ("effect descending, query ID, frozen condition order"),
            "cases": cases,
        },
        handle,
        indent=2,
        ensure_ascii=False,
    )
    handle.write("\n")

print(f"Selected {len(cases)} cases: {output}")

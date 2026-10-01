"""Prepare, review and freeze query variants for existing retrievers."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

from retrieval_lab.data import perturbations as p
from retrieval_lab.data.loader import load_dataset


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def candidates(
    data_dir: Path,
    qrels: Path,
    paraphrase_count: int = 0,
) -> dict:
    if type(paraphrase_count) is not int or not 0 <= paraphrase_count <= 50:
        raise ValueError("paraphrase_count must be an integer from 0 to 50")

    paths = {
        "corpus": data_dir / "corpus.jsonl",
        "queries": data_dir / "queries.jsonl",
        "qrels": qrels,
    }

    inputs = {name: digest(path) for name, path in paths.items()}

    dataset = load_dataset(data_dir, qrels)

    rows = [
        row
        for query_id, text in sorted(dataset.queries.items())
        for row in p.typo_variants(query_id, text)
    ]

    for condition in ("typo_light", "typo_heavy"):
        eligible = [
            row
            for row in rows
            if row["condition"] == condition and row["status"] != "ineligible"
        ]

        sample = set(
            p.select_ids(
                [row["parent_query_id"] for row in eligible],
                30,
                f"audit:{condition}",
            )
        )

        for row in rows:
            if row["condition"] == condition:
                row["review_required"] = row["status"] != "ineligible" and (
                    row["parent_query_id"] in sample or row["flagged"]
                )

    selected = p.select_ids(
        list(dataset.queries),
        paraphrase_count,
        "paraphrase",
    )

    for query_id in selected:
        rows.append(
            {
                "variant_id": (f"{query_id}::paraphrase::{p.VERSION}"),
                "parent_query_id": query_id,
                "condition": "paraphrase",
                "original_text": dataset.queries[query_id],
                "text": "",
                "seed": p.stable_seed("paraphrase"),
                "status": "pending",
                "review_note": "",
                "review_required": True,
            }
        )

    if inputs != {name: digest(path) for name, path in paths.items()}:
        raise ValueError("Source files changed during generation")

    return {
        "metadata": {
            "schema_version": 1,
            "transformation_version": p.VERSION,
            "seed": p.SEED,
            "inputs": inputs,
            "code_sha256": {
                "perturbations": digest(Path(p.__file__)),
                "perturbation_runs": digest(Path(__file__)),
            },
            "paraphrase_count": paraphrase_count,
            "paraphrase_ids": selected,
            "typo_audit_sample_per_severity": 30,
        },
        "variants": rows,
    }


def resolve_reviews(
    expected: dict,
    reviewed: dict,
) -> list[dict]:
    if reviewed["metadata"] != expected["metadata"]:
        raise ValueError("Source, code or selection metadata changed")

    supplied = reviewed["variants"]
    by_id = {row["variant_id"]: row for row in supplied}

    if len(by_id) != len(supplied) or set(by_id) != {
        row["variant_id"] for row in expected["variants"]
    }:
        raise ValueError("Variant IDs are missing, duplicated or unexpected")

    rows = []

    for original in expected["variants"]:
        row = dict(by_id[original["variant_id"]])

        mutable = {"status", "review_note"}

        if original["condition"] == "paraphrase":
            mutable.add("text")

        if {key: value for key, value in row.items() if key not in mutable} != {
            key: value for key, value in original.items() if key not in mutable
        }:
            raise ValueError("Generated variant content was edited")

        if original["status"] == "ineligible":
            if row != original:
                raise ValueError("Ineligible variants must remain unchanged")
        elif row["status"] not in {
            "pending",
            "accepted",
            "rejected",
        }:
            raise ValueError("Invalid review status")

        if row["status"] in {"accepted", "rejected"}:
            note = row["review_note"]

            if not isinstance(note, str) or not note.strip():
                raise ValueError("Review decisions need a note")

        if row["condition"] == "paraphrase" and row["status"] == "accepted":
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("An accepted paraphrase needs text")

            clean = " ".join(row["original_text"].casefold().split())
            changed = " ".join(row["text"].casefold().split())

            if changed == clean:
                raise ValueError("An unchanged query is not a paraphrase")

        rows.append(row)

    for condition in (
        "typo_light",
        "typo_heavy",
        "paraphrase",
    ):
        group = [
            row
            for row in rows
            if row["condition"] == condition and row["status"] != "ineligible"
        ]

        full_review = condition == "paraphrase" or any(
            row["status"] == "rejected" for row in group
        )

        for row in group:
            if row["status"] == "pending":
                if full_review or row["review_required"]:
                    raise ValueError(f"Review incomplete for {condition}")

                row["status"] = "accepted_by_sample_audit"
                row["review_note"] = "Not individually reviewed; required audit passed"

    return rows


def freeze_review(
    review: Path,
    data_dir: Path,
    qrels: Path,
    output: Path,
) -> None:
    raw = review.read_bytes()
    reviewed = json.loads(raw)

    expected = candidates(
        data_dir,
        qrels,
        reviewed["metadata"]["paraphrase_count"],
    )

    rows = resolve_reviews(expected, reviewed)
    dataset = load_dataset(data_dir, qrels)

    accepted = {
        "accepted",
        "accepted_by_sample_audit",
    }

    groups = {
        "clean": dict(dataset.queries),
    }

    for condition in (
        "typo_light",
        "typo_heavy",
        "paraphrase",
    ):
        groups[condition] = {
            row["parent_query_id"]: row["text"]
            for row in rows
            if row["condition"] == condition and row["status"] in accepted
        }

    # Review checks happen before creating output.
    # An existing freeze is never overwritten.
    output.mkdir(parents=True, exist_ok=False)

    files = {}

    for condition, queries in groups.items():
        if not queries:
            continue

        folder = output / condition
        folder.mkdir()

        shutil.copyfile(
            data_dir / "corpus.jsonl",
            folder / "corpus.jsonl",
        )

        with (folder / "queries.jsonl").open(
            "x",
            encoding="utf-8",
        ) as handle:
            for query_id, text in sorted(queries.items()):
                handle.write(
                    json.dumps(
                        {
                            "_id": query_id,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        with (folder / "qrels.tsv").open(
            "x",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle, delimiter="\t")

            writer.writerow(["query-id", "corpus-id", "score"])

            for query_id in sorted(queries):
                for doc_id, grade in sorted(dataset.qrels[query_id].items()):
                    writer.writerow([query_id, doc_id, grade])

        load_dataset(folder, folder / "qrels.tsv")

        for filename in (
            "corpus.jsonl",
            "queries.jsonl",
            "qrels.tsv",
        ):
            files[f"{condition}/{filename}"] = digest(folder / filename)

        if (
            files[f"{condition}/corpus.jsonl"]
            != expected["metadata"]["inputs"]["corpus"]
        ):
            raise ValueError("Corpus changed while exporting")

    current = {
        "corpus": digest(data_dir / "corpus.jsonl"),
        "queries": digest(data_dir / "queries.jsonl"),
        "qrels": digest(qrels),
    }

    if current != expected["metadata"]["inputs"]:
        raise ValueError("Inputs changed while freezing")

    manifest = {
        "metadata": expected["metadata"],
        "review_sha256": hashlib.sha256(raw).hexdigest(),
        "variants": rows,
        "accepted_counts": {name: len(group) for name, group in groups.items()},
        "files": files,
        "paraphrases_quantitative": (len(groups["paraphrase"]) >= 30),
    }

    write_json(output / "manifest.json", manifest)

    (output / "manifest.sha256").write_text(
        digest(output / "manifest.json") + "\n",
        encoding="utf-8",
    )

import copy
import json
from pathlib import Path

import pytest

from retrieval_lab.data.perturbations import (
    PROTECTED,
    typo_variants,
)
from retrieval_lab.perturbation_runs import (
    candidates,
    digest,
    freeze_review,
    resolve_reviews,
    write_json,
)

TEXT = "BRCA1 regulates cellular responses without increasing IL-6 at 20 percent."


def test_nested_edits_and_protected_content() -> None:
    light, heavy = typo_variants("q1", TEXT)

    assert heavy["edits"][:1] == light["edits"]
    assert light["edit_count"] == 1
    assert heavy["edit_count"] == 2

    assert len({edit["start"] for edit in heavy["edits"]}) == 2

    for row in (light, heavy):
        assert len(row["text"]) == len(TEXT)

        for edit in row["edits"]:
            before = edit["before"]
            after = edit["after"]

            assert before not in PROTECTED
            assert after[0] == before[0]
            assert after[-1] == before[-1]
            assert sorted(after) == sorted(before)

            differences = [
                i for i, (left, right) in enumerate(zip(before, after)) if left != right
            ]

            assert len(differences) == 2
            assert differences[1] == differences[0] + 1

        for protected in (
            "BRCA1",
            "without",
            "increasing",
            "IL-6",
            "20",
        ):
            assert protected in row["text"]


def test_generation_is_independent_of_other_queries() -> None:
    first = typo_variants("q1", TEXT)

    typo_variants(
        "unrelated",
        "another cellular response",
    )

    assert typo_variants("q1", TEXT) == first


def test_ineligible_is_not_a_successful_noop() -> None:
    light, heavy = typo_variants("q1", "protein")

    assert light["status"] == "pending"
    assert heavy["status"] == "ineligible"
    assert heavy["edit_count"] == 0

    assert all(
        row["status"] == "ineligible"
        for row in typo_variants(
            "q2",
            "not IL-6 20 aaaaaa",
        )
    )


@pytest.fixture
def sources(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source"
    root.mkdir()

    (root / "corpus.jsonl").write_text(
        json.dumps(
            {
                "_id": "d1",
                "title": "Protein",
                "text": "Cellular response.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with (root / "queries.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for i in range(35):
            handle.write(
                json.dumps(
                    {
                        "_id": f"q{i:02}",
                        "text": TEXT,
                    }
                )
                + "\n"
            )

    qrels = root / "qrels.tsv"
    qrels.write_text(
        "query-id\tcorpus-id\tscore\n" + "".join(f"q{i:02}\td1\t1\n" for i in range(35)),
        encoding="utf-8",
    )

    return root, qrels


def accept_required_for_test(bundle: dict) -> None:
    for row in bundle["variants"]:
        if row["review_required"]:
            row["status"] = "accepted"
            row["review_note"] = "Synthetic test decision"


def test_audit_gate_and_full_review_escalation(
    sources: tuple[Path, Path],
) -> None:
    root, qrels = sources
    expected = candidates(root, qrels)
    reviewed = copy.deepcopy(expected)

    with pytest.raises(ValueError, match="incomplete"):
        resolve_reviews(expected, reviewed)

    accept_required_for_test(reviewed)
    resolved = resolve_reviews(expected, reviewed)

    assert sum(row["status"] == "accepted_by_sample_audit" for row in resolved) == 10

    rejected = next(row for row in reviewed["variants"] if row["review_required"])
    rejected["status"] = "rejected"

    with pytest.raises(ValueError, match="incomplete"):
        resolve_reviews(expected, reviewed)


def test_modified_typo_text_is_rejected(
    sources: tuple[Path, Path],
) -> None:
    root, qrels = sources
    expected = candidates(root, qrels)
    reviewed = copy.deepcopy(expected)

    reviewed["variants"][0]["text"] = "different claim"

    with pytest.raises(ValueError, match="edited"):
        resolve_reviews(expected, reviewed)


def test_export_preserves_data_and_refuses_overwrite(
    sources: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    root, qrels = sources
    bundle = candidates(root, qrels)
    accept_required_for_test(bundle)

    review = tmp_path / "review.json"
    write_json(review, bundle)

    output = tmp_path / "frozen"
    freeze_review(review, root, qrels, output)

    assert (output / "clean/qrels.tsv").read_bytes() == (
        output / "typo_light/qrels.tsv"
    ).read_bytes()

    assert digest(output / "typo_heavy/corpus.jsonl") == digest(root / "corpus.jsonl")

    assert (output / "manifest.sha256").read_text().strip() == digest(
        output / "manifest.json"
    )

    with pytest.raises(FileExistsError):
        freeze_review(review, root, qrels, output)


def test_unchanged_paraphrase_is_rejected(
    sources: tuple[Path, Path],
) -> None:
    root, qrels = sources
    expected = candidates(root, qrels, paraphrase_count=1)
    reviewed = copy.deepcopy(expected)
    accept_required_for_test(reviewed)

    row = next(row for row in reviewed["variants"] if row["condition"] == "paraphrase")
    row["text"] = row["original_text"]

    with pytest.raises(ValueError, match="unchanged"):
        resolve_reviews(expected, reviewed)

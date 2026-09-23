import hashlib
import json
from pathlib import Path

import pytest

from retrieval_lab.config import RunConfig
from retrieval_lab.hybrid_runs import build_hybrid_run


@pytest.fixture
def source_runs(tmp_path: Path) -> tuple[Path, Path]:
    paths = []

    for method, ids in (
        ("bm25", ["d01", "d02", "d03"]),
        ("dense", ["d02", "d04", "d01"]),
    ):
        run = {
            "schema_version": 1,
            "metadata": {
                "method": method,
                "config": {"seed": 42, "top_k": 3},
                "documents": 4,
                "queries": 2,
                "inputs": {
                    name: {
                        "path": f"{name}.fixture",
                        "sha256": "a" * 64,
                    }
                    for name in ("corpus", "queries", "qrels")
                },
            },
            "results": {
                "q01": [
                    {
                        "rank": rank,
                        "document_id": doc_id,
                        "score": float(4 - rank),
                    }
                    for rank, doc_id in enumerate(ids, start=1)
                ],
                "q02": [],
            },
        }

        path = tmp_path / f"{method}.json"
        path.write_text(json.dumps(run), encoding="utf-8")
        paths.append(path)

    return paths[0], paths[1]


def test_hybrid_run_preserves_structure_and_provenance(
    source_runs: tuple[Path, Path],
) -> None:
    bm25, dense = source_runs

    run = build_hybrid_run(
        bm25,
        dense,
        RunConfig(seed=42, top_k=3),
    )

    assert run["metadata"]["method"] == "hybrid_rrf"
    assert run["metadata"]["config"]["top_k"] == 3
    assert run["metadata"]["fusion"]["rrf_k"] == 60
    assert run["results"]["q02"] == []

    assert [hit["document_id"] for hit in run["results"]["q01"]] == ["d02", "d01", "d04"]

    assert run["metadata"]["source_runs"]["bm25"]["sha256"] == (
        hashlib.sha256(bm25.read_bytes()).hexdigest()
    )


def test_raw_score_scale_does_not_change_fusion(
    source_runs: tuple[Path, Path],
) -> None:
    bm25, dense = source_runs
    settings = RunConfig(seed=42, top_k=3)

    before = build_hybrid_run(bm25, dense, settings)

    source = json.loads(dense.read_text(encoding="utf-8"))
    for hit in source["results"]["q01"]:
        hit["score"] *= 1000
    dense.write_text(json.dumps(source), encoding="utf-8")

    after = build_hybrid_run(bm25, dense, settings)

    assert before["results"] == after["results"]


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("corpus", "hash mismatch"),
        ("queries", "hash mismatch"),
        ("qrels", "hash mismatch"),
        ("missing_query", "query IDs"),
        ("depth", "retrieval depths"),
        ("duplicate", "Duplicate"),
        ("rank", "stored rank"),
    ],
)
def test_incompatible_sources_are_rejected(
    source_runs: tuple[Path, Path],
    problem: str,
    message: str,
) -> None:
    bm25, dense = source_runs
    source = json.loads(dense.read_text(encoding="utf-8"))

    if problem in {"corpus", "queries", "qrels"}:
        source["metadata"]["inputs"][problem]["sha256"] = "b" * 64
    elif problem == "missing_query":
        del source["results"]["q02"]
        source["metadata"]["queries"] = 1
    elif problem == "depth":
        source["metadata"]["config"]["top_k"] = 4
    elif problem == "duplicate":
        source["results"]["q01"][1]["document_id"] = "d02"
    elif problem == "rank":
        source["results"]["q01"][0]["rank"] = 0

    dense.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_hybrid_run(
            bm25,
            dense,
            RunConfig(seed=42, top_k=3),
        )


def test_output_depth_cannot_exceed_source_depth(
    source_runs: tuple[Path, Path],
) -> None:
    bm25, dense = source_runs

    with pytest.raises(ValueError, match="exceeds"):
        build_hybrid_run(
            bm25,
            dense,
            RunConfig(seed=42, top_k=4),
        )


def test_cli_saves_run_and_refuses_overwrite(
    source_runs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from retrieval_lab.cli import app

    bm25, dense = source_runs
    config = tmp_path / "config.yaml"
    config.write_text("seed: 42\ntop_k: 3\n", encoding="utf-8")
    output = tmp_path / "hybrid.json"

    args = [
        "fuse-rrf",
        "--bm25-run",
        str(bm25),
        "--dense-run",
        str(dense),
        "--config",
        str(config),
        "--output",
        str(output),
    ]

    runner = CliRunner()
    first = runner.invoke(app, args)

    assert first.exit_code == 0, first.output

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["metadata"]["method"] == "hybrid_rrf"

    original = output.read_bytes()
    repeated = runner.invoke(app, args)

    assert repeated.exit_code == 1
    assert "already exists" in repeated.output
    assert output.read_bytes() == original

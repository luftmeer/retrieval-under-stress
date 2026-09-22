import json
from pathlib import Path

from typer.testing import CliRunner

from retrieval_lab.cli import app

runner = CliRunner()


def test_help_list_doctor() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.output


def test_doctor_accepts_valid_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 42\ntop_k: 10\n", encoding="utf-8")

    result = runner.invoke(app, ["doctor", "--config", str(path)])

    assert result.exit_code == 0
    assert "seed=42, top_k=10" in result.output


def test_doctor_rejects_invalid_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("top_k: 0\n", encoding="utf-8")

    result = runner.invoke(app, ["doctor", "--config", str(path)])

    assert result.exit_code == 1
    assert "top_k" in result.output


def test_doctor_reports_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"

    result = runner.invoke(app, ["doctor", "--config", str(path)])

    assert result.exit_code == 1
    assert "Configuration error" in result.output


def test_data_check_accepts_fixtures() -> None:
    fixtures = Path(__file__).parent / "fixtures"

    result = runner.invoke(
        app,
        [
            "data-check",
            "--data-dir",
            str(fixtures),
            "--qrels",
            str(fixtures / "qrels.tsv"),
        ],
    )

    assert result.exit_code == 0
    assert "Documents: 20" in result.output
    assert "Evaluation queries: 5" in result.output
    assert "Judgments: 8" in result.output


def test_data_check_reports_missing_dataset(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "data-check",
            "--data-dir",
            str(tmp_path / "missing"),
        ],
    )

    assert result.exit_code == 1
    assert "Dataset error:" in result.output


def test_bm25_command_saves_reproducible_run(
    tmp_path: Path,
) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    config = tmp_path / "config.yaml"
    config.write_text("seed: 42\ntop_k: 3\n", encoding="utf-8")

    common_args = [
        "retrieve-bm25",
        "--data-dir",
        str(fixtures),
        "--qrels",
        str(fixtures / "qrels.tsv"),
        "--config",
        str(config),
    ]

    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    first = runner.invoke(
        app,
        [*common_args, "--output", str(first_output)],
    )

    second = runner.invoke(
        app,
        [*common_args, "--output", str(second_output)],
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert first_output.read_bytes() == second_output.read_bytes()

    run = json.loads(first_output.read_text(encoding="utf-8"))

    assert run["schema_version"] == 1
    assert run["metadata"]["documents"] == 20
    assert run["metadata"]["queries"] == 5
    assert run["metadata"]["config"]["top_k"] == 3

    assert set(run["results"]) == {
        "q01",
        "q02",
        "q03",
        "q04",
        "q05",
    }

    for hits in run["results"].values():
        assert [hit["rank"] for hit in hits] == [1, 2, 3]
        assert len({hit["document_id"] for hit in hits}) == 3

        scores = [hit["score"] for hit in hits]
        assert scores == sorted(scores, reverse=True)


def test_bm25_command_preserves_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("original content\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["retrieve-bm25", "--output", str(output)],
    )

    assert result.exit_code == 1
    assert "already exists" in result.output
    assert output.read_text(encoding="utf-8") == "original content\n"


def test_bm25_command_reports_missing_dataset(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("top_k: 3\n", encoding="utf-8")
    output = tmp_path / "run.json"

    result = runner.invoke(
        app,
        [
            "retrieve-bm25",
            "--data-dir",
            str(tmp_path / "missing"),
            "--config",
            str(config),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert "Retrieval error:" in result.output
    assert not output.exists()


def test_evaluate_command_saves_report(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    config = tmp_path / "config.yaml"
    config.write_text("seed: 42\ntop_k: 3\n", encoding="utf-8")

    run_path = tmp_path / "run.json"
    report_path = tmp_path / "evaluation.json"

    retrieval = runner.invoke(
        app,
        [
            "retrieve-bm25",
            "--data-dir",
            str(fixtures),
            "--qrels",
            str(fixtures / "qrels.tsv"),
            "--config",
            str(config),
            "--output",
            str(run_path),
        ],
    )

    assert retrieval.exit_code == 0, retrieval.output

    args = [
        "evaluate",
        "--run",
        str(run_path),
        "--data-dir",
        str(fixtures),
        "--qrels",
        str(fixtures / "qrels.tsv"),
        "--k",
        "3",
        "--output",
        str(report_path),
    ]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "mrr@3:" in result.output

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["metadata"]["queries"] == 5
    assert report["metadata"]["k"] == 3

    original = report_path.read_bytes()
    repeated = runner.invoke(app, args)

    assert repeated.exit_code == 1
    assert "already exists" in repeated.output
    assert report_path.read_bytes() == original

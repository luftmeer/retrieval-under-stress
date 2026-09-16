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

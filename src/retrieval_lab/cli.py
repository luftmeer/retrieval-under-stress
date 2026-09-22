"""Terminal commands for the application"""

from pathlib import Path
from typing import Annotated
from zipfile import BadZipFile

import typer
import yaml

from retrieval_lab.config import load_config
from retrieval_lab.data.download import download_scifact
from retrieval_lab.data.loader import load_dataset
from retrieval_lab.runs import build_bm25_run, save_run

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Retrieval Under Stress: reproducible retrieval experiments."""


@app.command()
def doctor(
    config: Annotated[
        Path,
        typer.Option(help="Path to a YAML configuration file."),
    ] = Path("configs/smoke.yaml"),
) -> None:
    """Validate configuration without downloading data or starting services."""

    try:
        settings = load_config(config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Configuration OK")
    typer.echo(f"seed={settings.seed}, top_k={settings.top_k}")


@app.command("data-download")
def data_download(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for downloaded datasets."),
    ] = Path("data/raw"),
) -> None:
    """Download and validate the BEIR SciFact test dataset."""
    try:
        destination = download_scifact(output_dir)
    except (OSError, ValueError, BadZipFile) as exc:
        typer.echo(f"Download error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Dataset ready: {destination}")
    typer.echo(f"Provenance: {destination / 'manifest.json'}")


@app.command("data-check")
def data_check(
    data_dir: Annotated[
        Path, typer.Option(help="Directory containing corpus.jsonl and queries.jsonl.")
    ] = Path("data/raw/scifact"),
    split: Annotated[
        str,
        typer.Option(help="Qrels split: train, dev, or test."),
    ] = "test",
    qrels: Annotated[
        Path | None, typer.Option(help="Explicit qrels path; overrides --split")
    ] = None,
) -> None:
    """Validate local dataset files and report selected-split counts."""
    try:
        if qrels is None:
            if split not in {"train", "dev", "test"}:
                raise ValueError("split must be train, dev or test")

            qrels_path = data_dir / "qrels" / f"{split}.tsv"

        else:
            qrels_path = qrels

        dataset = load_dataset(data_dir, qrels_path)

    except (OSError, ValueError) as exc:
        typer.echo(f"Dataset error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    judgment_count = sum(len(rows) for rows in dataset.qrels.values())

    typer.echo("Dataset OK")
    typer.echo(f"Qrels: {qrels_path}")
    typer.echo(f"Documents: {len(dataset.documents)}")
    typer.echo(f"Evaluation queries: {len(dataset.queries)}")
    typer.echo(f"Judgments: {judgment_count}")


@app.command("retrieve-bm25")
def retrieve_bm25(
    data_dir: Annotated[
        Path,
        typer.Option(help="Directory containing corpus.jsonl and queries.jsonl"),
    ] = Path("data/raw/scifact"),
    config: Annotated[
        Path, typer.Option(help="Path to a YAML configuration file.")
    ] = Path("configs/smoke.yaml"),
    qrels: Annotated[
        Path | None,
        typer.Option(help="Qrels path; defaults to DATA_DIR/qrels/test.tsv"),
    ] = None,
    output: Annotated[
        Path, typer.Option(help="New JSON output file; existing files are rejected.")
    ] = Path("output/bm25.json"),
) -> None:
    """Rank selected queries with BM25 and save the retrieval run."""
    try:
        if output.exists():
            raise FileExistsError(
                f"{output} already exists; choose another --output path"
            )

        settings = load_config(config)
        qrels_path = qrels if qrels is not None else data_dir / "qrels" / "test.tsv"

        run = build_bm25_run(data_dir, qrels_path, settings)
        save_run(run, output)

    except (OSError, ValueError, yaml.YAMLError) as exc:
        typer.echo(f"Retrieval error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("BM25 run saved")
    typer.echo(f"Output: {output}")

"""Terminal commands for the application"""

from pathlib import Path
from typing import Annotated

import typer
import yaml

from retrieval_lab.config import load_config

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

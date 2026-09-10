"""Load and validate experiment settings."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, strict=True, ge=0)
    top_k: int = Field(default=10, strict=True, ge=1, le=100)


def load_config(path: Path) -> RunConfig:
    """Read YAML and validate its settings."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            "Configuration must be a YAML mapping of setting names to values."
        )

    return RunConfig.model_validate(data)

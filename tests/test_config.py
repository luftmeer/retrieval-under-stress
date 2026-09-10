from pathlib import Path

import pytest
from pydantic import ValidationError

from retrieval_lab.config import load_config


def test_valid_configuration(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 7\ntop_k: 5\n", encoding="utf-8")

    config = load_config(path)

    assert config.seed == 7
    assert config.top_k == 5


@pytest.mark.parametrize(
    "content",
    [
        "top_k: 0",
        "top_k: 101",
        "topk: 10",
        'top_k: "10"',
    ],
)
def test_invalid_settings_are_rejected(
    tmp_path: Path,
    content: str,
) -> None:

    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize("content", ["", "- 10\n- 20"])
def test_non_mapping_yaml_is_rejected(
    tmp_path: Path,
    content: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)

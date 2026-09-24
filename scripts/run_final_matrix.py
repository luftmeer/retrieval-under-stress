"""Run from the repository root; supply a new results directory."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from retrieval_lab.robustness import METHODS, sha256, verify_freeze

root = Path("data/processed/stress-test-v1")
config = Path("configs/retrieval.yaml")
output = Path(sys.argv[1])

frozen = verify_freeze(root)
settings = yaml.safe_load(config.read_text())

if settings["top_k"] != 100 or settings["seed"] != 42:
    raise ValueError("Expected top_k=100 and seed=42")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprints() -> dict:
    paths = [
        config,
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    paths += list(Path("src/retrieval_lab").rglob("*.py"))
    paths += list(Path("scripts").glob("*.py"))

    return {str(path): sha256(path) for path in sorted(paths)}


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
    ).strip()


state = {
    "status": "running",
    "started_at": now(),
    "git_commit": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain")),
    "python": sys.version,
    "platform": platform.platform(),
    "config": settings,
    "freeze_sha256": sha256(root / "manifest.json"),
    "accepted_counts": frozen["accepted_counts"],
    "source_sha256": fingerprints(),
    "commands": [],
}

output.mkdir(parents=True, exist_ok=False)


def save_state() -> None:
    (output / "manifest.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def execute(command: str, *arguments: object) -> None:
    argv = ["uv", "run", "--frozen", "retrieval-lab", command]
    argv += [str(value) for value in arguments]

    state["commands"].append(argv)
    save_state()

    subprocess.run(argv, check=True)


save_state()

try:
    for condition, n in frozen["accepted_counts"].items():
        if n == 0:
            print(
                f"Unavailable: {condition} (no accepted variants)",
                flush=True,
            )
            continue

        data = root / condition
        common = (
            "--data-dir",
            data,
            "--qrels",
            data / "qrels.tsv",
        )

        for method in ("bm25", "dense"):
            execute(
                f"retrieve-{method}",
                *common,
                "--config",
                config,
                "--output",
                output / f"{condition}-{method}.json",
            )

        execute(
            "fuse-rrf",
            "--bm25-run",
            output / f"{condition}-bm25.json",
            "--dense-run",
            output / f"{condition}-dense.json",
            "--config",
            config,
            "--rrf-k",
            60,
            "--output",
            output / f"{condition}-hybrid.json",
        )

        for method in METHODS:
            execute(
                "evaluate",
                *common,
                "--run",
                output / f"{condition}-{method}.json",
                "--k",
                10,
                "--output",
                output / f"{condition}-{method}-evaluation.json",
            )

    verify_freeze(root)

    if fingerprints() != state["source_sha256"]:
        raise ValueError("Code, configuration or dependency files changed during run")

    state["output_sha256"] = {
        path.name: sha256(path)
        for path in sorted(output.glob("*.json"))
        if path.name != "manifest.json"
    }
    state["status"] = "complete"

except BaseException as exc:
    state["status"] = "failed"
    state["error"] = repr(exc)
    raise

finally:
    state["finished_at"] = now()
    save_state()

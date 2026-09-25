"""Reproduce the complete Retrieval Under Stress evidence pipeline."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=ROOT,
        text=True,
    ).strip()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_complete(path: Path) -> dict:
    record = read_json(path)

    if record.get("status") != "complete":
        raise ValueError(f"Incomplete stage: {path}")

    return record


def validate_latency_environment(path: Path) -> None:
    environment = read_json(path)

    required = (
        "cpu",
        "ram_gib",
        "qdrant_runtime",
    )

    missing = [key for key in required if not environment.get(key)]

    if missing:
        raise ValueError(f"Incomplete latency environment: {missing}")


def check_qdrant() -> None:
    command = [
        "uv",
        "run",
        "--locked",
        "python",
        "-c",
        (
            "from qdrant_client import QdrantClient; "
            "client = QdrantClient("
            "url='http://localhost:6333', timeout=10"
            "); "
            "print('Qdrant:', client.info().version); "
            "client.close()"
        ),
    ]

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def source_fingerprints(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def run_stage(
    name: str,
    command: list[str],
    *,
    state: dict,
    state_path: Path,
    logs: Path,
) -> None:
    log_path = logs / f"{name}.log"

    print()
    print(f"=== {name} ===")
    print(" ".join(command))
    print()

    started = now()

    with log_path.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        try:
            for line in process.stdout:
                print(line, end="")
                handle.write(line)

        except KeyboardInterrupt:
            process.terminate()
            process.wait()

            state["stages"].append(
                {
                    "name": name,
                    "status": "interrupted",
                    "started_at": started,
                    "finished_at": now(),
                    "returncode": process.returncode,
                    "command": command,
                    "log": str(log_path),
                }
            )

            save_state(state_path, state)
            raise

        returncode = process.wait()

    stage = {
        "name": name,
        "status": ("complete" if returncode == 0 else "failed"),
        "started_at": started,
        "finished_at": now(),
        "returncode": returncode,
        "command": command,
        "log": str(log_path),
        "log_sha256": sha256(log_path),
    }

    state["stages"].append(stage)
    save_state(state_path, state)

    if returncode != 0:
        raise RuntimeError(f"Stage failed: {name}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "output",
        type=Path,
        help="New reproduction directory.",
    )

    parser.add_argument(
        "--latency-environment",
        type=Path,
        default=Path("configs/latency-environment.json"),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without running them.",
    )

    args = parser.parse_args()

    output = args.output.resolve()

    quality = output / "quality"
    latency = output / "latency"
    figures = output / "figures"
    logs = output / "logs"
    receipt = output / "mlflow-receipt.json"

    required = [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "configs/retrieval.yaml",
        ROOT / "configs/dense-model.json",
        ROOT / args.latency_environment,
        ROOT / "data/manifests/stress-test-freeze-v1.json",
        ROOT / "data/manifests/stress-test-freeze-v1.sha256",
        ROOT / "scripts/run_final_matrix.py",
        ROOT / "scripts/analyze_robustness.py",
        ROOT / "scripts/select_cases.py",
        ROOT / "scripts/measure_latency.py",
        ROOT / "scripts/make_figures.py",
        ROOT / "scripts/make_figures.py.lock",
        ROOT / "scripts/import_mlflow.py",
        ROOT / "tools/tracking/pyproject.toml",
        ROOT / "tools/tracking/uv.lock",
    ]

    for path in required:
        require_file(path)

    validate_latency_environment(ROOT / args.latency_environment)

    commands = [
        (
            "retrieval",
            [
                "uv",
                "run",
                "--locked",
                "python",
                "scripts/run_final_matrix.py",
                str(quality),
            ],
        ),
        (
            "analysis",
            [
                "uv",
                "run",
                "--locked",
                "python",
                "scripts/analyze_robustness.py",
                str(quality),
            ],
        ),
        (
            "case-selection",
            [
                "uv",
                "run",
                "--locked",
                "python",
                "scripts/select_cases.py",
                str(quality),
            ],
        ),
        (
            "latency",
            [
                "uv",
                "run",
                "--locked",
                "python",
                "scripts/measure_latency.py",
                str(quality),
                str(latency),
                str(ROOT / args.latency_environment),
            ],
        ),
        (
            "figures",
            [
                "uv",
                "run",
                "--locked",
                "--script",
                "scripts/make_figures.py",
                str(quality),
                str(latency),
                str(figures),
            ],
        ),
        (
            "mlflow-import",
            [
                "uv",
                "run",
                "--project",
                "tools/tracking",
                "--locked",
                "python",
                "scripts/import_mlflow.py",
                str(quality),
                str(latency),
                str(figures),
                "--name",
                f"reproduction-{output.name}",
                "--receipt",
                str(receipt),
            ],
        ),
    ]

    if args.dry_run:
        for name, command in commands:
            print(f"\n{name}:")
            print(" ".join(command))

        return

    if git("status", "--porcelain"):
        raise ValueError("Working tree is not clean; commit or revert changes first")

    check_qdrant()

    output.mkdir(parents=True, exist_ok=False)
    logs.mkdir()

    tracked_sources = {
        *required,
        ROOT / "scripts/reproduce.py",
        *ROOT.joinpath("src/retrieval_lab").rglob("*.py"),
    }

    state = {
        "status": "running",
        "started_at": now(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": False,
        "python": sys.version,
        "platform": platform.platform(),
        "latency_environment": str(ROOT / args.latency_environment),
        "source_sha256": source_fingerprints(list(tracked_sources)),
        "stages": [],
    }

    state_path = output / "manifest.json"
    save_state(state_path, state)

    try:
        for name, command in commands:
            run_stage(
                name,
                command,
                state=state,
                state_path=state_path,
                logs=logs,
            )

            if name == "retrieval":
                require_complete(quality / "manifest.json")

            elif name == "analysis":
                require_complete(quality / "analysis/manifest.json")

            elif name == "case-selection":
                cases = read_json(quality / "case-selection.json")["cases"]

                if len(cases) != 6:
                    raise ValueError("Expected six diagnostic cases")

            elif name == "latency":
                require_complete(latency / "manifest.json")

            elif name == "figures":
                require_complete(figures / "manifest.json")

            elif name == "mlflow-import":
                imported = require_complete(receipt)

                if imported["retrieval_manifest_sha256"] != sha256(
                    quality / "manifest.json"
                ):
                    raise ValueError("MLflow receipt refers to another run")

        if source_fingerprints(list(tracked_sources)) != state["source_sha256"]:
            raise ValueError("Source files changed during reproduction")

        state["evidence_sha256"] = {
            "quality": sha256(quality / "manifest.json"),
            "analysis": sha256(quality / "analysis/manifest.json"),
            "latency": sha256(latency / "manifest.json"),
            "figures": sha256(figures / "manifest.json"),
            "mlflow_receipt": sha256(receipt),
        }

        state["status"] = "complete"

    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        raise

    finally:
        state["finished_at"] = now()
        save_state(state_path, state)


if __name__ == "__main__":
    main()

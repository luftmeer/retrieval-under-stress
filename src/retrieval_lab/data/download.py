"""Download the BEIR SciFact archive and record its provenance."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

from retrieval_lab.data.loader import load_dataset

SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)

REQUIRED_FILES = (
    "corpus.jsonl",
    "queries.jsonl",
    "qrels/test.tsv",
)


def sha256_file(path: Path) -> str:
    """Calculate a fingerprint without reading the whole file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def download_scifact(output_dir: Path) -> Path:
    """Download, validate and publish a local SciFact test dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "scifact"

    if destination.exists():
        raise FileExistsError(f"{destination} already exists; run data check on it.")

    with TemporaryDirectory(dir=output_dir) as temporary:
        staged = Path(temporary) / "scifact"
        staged.mkdir()
        archive_path = staged / "source.zip"

        with urlopen(SCIFACT_URL, timeout=60) as response:
            with archive_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)

        downloaded_at = datetime.now(timezone.utc).isoformat()

        with ZipFile(archive_path) as archive:
            for relative_path in REQUIRED_FILES:
                member = f"scifact/{relative_path}"

                if archive.namelist().count(member) != 1:
                    raise ValueError(f"Archive must contain exactly one {member}")

                target = staged / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)

                with archive.open(member) as source:
                    with target.open("wb") as handle:
                        shutil.copyfileobj(source, handle)

        dataset = load_dataset(staged, staged / "qrels/test.tsv")

        manifest = {
            "dataset": "beir/scifact",
            "source_url": SCIFACT_URL,
            "downloaded_at_utc": downloaded_at,
            "split": "test",
            "archive_sha256": sha256_file(archive_path),
            "file_sha256": {
                name: sha256_file(staged / name) for name in REQUIRED_FILES
            },
            "counts": {
                "documents": len(dataset.documents),
                "queries": len(dataset.queries),
                "judgments": sum(len(rows) for rows in dataset.qrels.values()),
            },
            "document_text_policy": (
                "title + newline + text; text alone when title is empty"
            ),
        }

        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        staged.rename(destination)

    return destination

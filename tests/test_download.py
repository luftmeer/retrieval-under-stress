import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from retrieval_lab.data import download

FIXTURES = Path(__file__).parent / "fixtures"


def test_download_extracts_validates_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = io.BytesIO()

    with ZipFile(buffer, "w") as archive:
        archive.write(FIXTURES / "corpus.jsonl", "scifact/corpus.jsonl")
        archive.write(FIXTURES / "queries.jsonl", "scifact/queries.jsonl")
        archive.write(FIXTURES / "qrels.tsv", "scifact/qrels/test.tsv")

    archive_bytes = buffer.getvalue()

    def fake_urlopen(url: str, timeout: int) -> io.BytesIO:
        assert url == download.SCIFACT_URL
        assert timeout > 0
        return io.BytesIO(archive_bytes)

    monkeypatch.setattr(download, "urlopen", fake_urlopen)

    destination = download.download_scifact(tmp_path / "raw")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["counts"] == {
        "documents": 20,
        "queries": 5,
        "judgments": 8,
    }

    assert manifest["split"] == "test"
    assert manifest["source_url"] == download.SCIFACT_URL
    assert manifest["archive_sha256"] == hashlib.sha256(archive_bytes).hexdigest()

    corpus_bytes = (FIXTURES / "corpus.jsonl").read_bytes()
    assert (destination / "corpus.jsonl").read_bytes() == corpus_bytes
    assert (
        manifest["file_sha256"]["corpus.jsonl"]
        == hashlib.sha256(corpus_bytes).hexdigest()
    )

    with pytest.raises(FileExistsError):
        download.download_scifact(tmp_path / "raw")


def test_incomplete_archive_does_not_publish_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.BytesIO()

    with ZipFile(buffer, "w") as archive:
        archive.writestr("unexpected.txt", "Missing dataset files")

    archive_bytes = buffer.getvalue()

    def fake_urlopen(url: str, timeout: int) -> io.BytesIO:
        return io.BytesIO(archive_bytes)

    monkeypatch.setattr(download, "urlopen", fake_urlopen)

    output_dir = tmp_path / "raw"

    with pytest.raises(ValueError, match="Archive must contain"):
        download.download_scifact(output_dir)

    assert not (output_dir / "scifact").exists()
    assert list(output_dir.iterdir()) == []

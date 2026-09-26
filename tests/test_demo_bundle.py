import hashlib
import json
from pathlib import Path


def test_committed_demo_bundle_is_valid():
    path = Path(
        "demo/benchmark-v1.json"
    )

    checksum_path = Path(
        "demo/benchmark-v1.sha256"
    )

    assert path.is_file()
    assert checksum_path.is_file()

    observed = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    expected = checksum_path.read_text(
        encoding="utf-8"
    ).strip()

    assert observed == expected

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert data["schema_version"] == 1

    assert data["methods"] == [
        "bm25",
        "dense",
        "hybrid",
    ]

    assert data["candidate_depth"] == 100
    assert data["rrf_k"] == 60
    assert data["max_rank"] == 20

    assert "clean" in data["conditions"]
    assert data["queries"]
    assert data["documents"]
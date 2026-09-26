"""Read and query the immutable benchmark demo bundle."""

import hashlib
import json
from pathlib import Path


METHOD_NAMES = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
}


CONDITION_NAMES = {
    "clean": "Clean",
    "typo_light": "Light typo",
    "typo_heavy": "Heavy typo",
    "paraphrase": "Paraphrase",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_bundle(path: Path) -> dict:
    checksum_path = path.with_suffix(
        ".sha256"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    if checksum_path.is_file():
        expected = checksum_path.read_text(
            encoding="utf-8"
        ).strip()

        observed = sha256(path)

        if observed != expected:
            raise ValueError(
                "Demo bundle checksum mismatch"
            )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if data.get("schema_version") != 1:
        raise ValueError(
            "Unsupported demo bundle schema"
        )

    if data.get("methods") != [
        "bm25",
        "dense",
        "hybrid",
    ]:
        raise ValueError(
            "Unexpected retrieval methods"
        )

    if "clean" not in data.get(
        "conditions",
        [],
    ):
        raise ValueError(
            "Demo bundle has no clean condition"
        )

    if not data.get("queries"):
        raise ValueError(
            "Demo bundle contains no queries"
        )

    return data


def stress_conditions(bundle: dict) -> list[str]:
    return [
        condition
        for condition in bundle["conditions"]
        if condition != "clean"
    ]


def query_ids_for_condition(
    bundle: dict,
    condition: str,
) -> list[str]:
    return [
        query_id
        for query_id, record
        in bundle["queries"].items()
        if condition in record["variants"]
    ]


def relevant(
    record: dict,
    document_id: str,
) -> bool:
    return (
        record["qrels"].get(
            document_id,
            0,
        )
        > 0
    )


def relevant_ranks(
    hits: list[dict],
    record: dict,
) -> list[int]:
    return [
        hit["rank"]
        for hit in hits
        if relevant(
            record,
            hit["document_id"],
        )
    ]
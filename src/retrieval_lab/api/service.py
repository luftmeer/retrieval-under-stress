"""Build and own resources shared by API requests."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from qdrant_client import QdrantClient

from retrieval_lab.data.download import sha256_file
from retrieval_lab.data.loader import Dataset, load_dataset
from retrieval_lab.dense_runs import (
    DIMENSION,
    MAX_SEQUENCE_LENGTH,
    MODEL_ID,
    TEXT_POLICY,
    load_encoder,
)
from retrieval_lab.retrieval.bm25 import BM25Retriever
from retrieval_lab.retrieval.dense import DenseIndex
from retrieval_lab.retrieval.live import SearchSession


@dataclass(frozen=True)
class ApiSettings:
    data_dir: Path = Path("data/processed/stress-test-v1/clean")
    model_path: Path = Path("configs/dense-model.json")
    qdrant_url: str = "http://localhost:6333"

    @property
    def qrels_path(self) -> Path:
        return self.data_dir / "qrels.tsv"


@dataclass
class ApiResources:
    dataset: Dataset
    session: SearchSession
    client: Any
    model_revision: str
    collection_name: str
    index_reused: bool
    search_lock: Lock = field(default_factory=Lock)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()


def collection_name(
    corpus_hash: str,
    revision: str,
) -> str:
    specification = {
        "corpus_sha256": corpus_hash,
        "model_id": MODEL_ID,
        "revision": revision,
        "dimension": DIMENSION,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "normalize_embeddings": True,
        "document_text_policy": TEXT_POLICY,
        "implementation": "api-index-v1",
    }

    payload = json.dumps(
        specification,
        sort_keys=True,
    ).encode("utf-8")

    fingerprint = hashlib.sha256(payload).hexdigest()

    return f"retrieval_lab_api_{fingerprint[:24]}"


def populate_index(
    index: DenseIndex,
    encoder,
    dataset: Dataset,
) -> None:
    document_ids = sorted(dataset.documents)

    texts = [dataset.documents[document_id].search_text for document_id in document_ids]

    vectors = encoder.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    index.upsert(
        document_ids,
        vectors,
    )


def build_resources(
    settings: ApiSettings,
) -> ApiResources:
    dataset = load_dataset(
        settings.data_dir,
        settings.qrels_path,
    )

    bm25 = BM25Retriever(
        dataset.documents,
    )

    encoder, revision = load_encoder(
        settings.model_path,
    )

    client = QdrantClient(
        url=settings.qdrant_url,
        timeout=60,
    )

    try:
        corpus_hash = sha256_file(settings.data_dir / "corpus.jsonl")

        name = collection_name(
            corpus_hash,
            revision,
        )

        existed = client.collection_exists(name)

        index = DenseIndex(
            client,
            name,
            dimension=DIMENSION,
        )

        reused = False

        if existed:
            count = client.count(
                collection_name=name,
                exact=True,
            ).count

            if count == len(dataset.documents):
                reused = True
            else:
                client.delete_collection(name)

                index = DenseIndex(
                    client,
                    name,
                    dimension=DIMENSION,
                )

        if not reused:
            populate_index(
                index,
                encoder,
                dataset,
            )

        session = SearchSession(
            bm25,
            encoder,
            index,
            candidate_depth=100,
        )

        return ApiResources(
            dataset=dataset,
            session=session,
            client=client,
            model_revision=revision,
            collection_name=name,
            index_reused=reused,
        )

    except BaseException:
        client.close()
        raise

"""Build dense retrieval runs compatible with the existing evaluator."""

import hashlib
import json
import re
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

from qdrant_client import QdrantClient

from retrieval_lab.config import RunConfig
from retrieval_lab.data.download import sha256_file
from retrieval_lab.data.loader import load_dataset
from retrieval_lab.retrieval.dense import DenseIndex

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQUENCE_LENGTH = 256
DIMENSION = 384

TEXT_POLICY = "title + newline + text; text alone when title is empty"


def load_encoder(model_path: Path):
    """Load one explicitly pinned model on CPU."""
    # Import lazily so ordinary CLI commands do not load PyTorch.
    from sentence_transformers import SentenceTransformer

    specification = json.loads(model_path.read_text(encoding="utf-8"))

    if not isinstance(specification, dict):
        raise ValueError("Model specification must be a JSON object")

    if specification.get("model_id") != MODEL_ID:
        raise ValueError("Unexpected embedding model")

    revision = specification.get("revision")

    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Model revision must be a full commit SHA")

    encoder = SentenceTransformer(
        MODEL_ID,
        revision=revision,
        device="cpu",
        trust_remote_code=False,
    )
    encoder.max_seq_length = MAX_SEQUENCE_LENGTH

    if encoder.get_sentence_embedding_dimension() != DIMENSION:
        raise ValueError("Unexpected embedding dimension")

    return encoder, revision


def truncated_ids(
    encoder,
    ids: list[str],
    texts: list[str],
) -> list[str]:
    """Identify inputs longer than the encoder's total token budget."""
    affected = []

    for start in range(0, len(texts), 64):
        batch = encoder.tokenizer(
            texts[start : start + 64],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]

        for offset, tokens in enumerate(batch):
            if len(tokens) > MAX_SEQUENCE_LENGTH:
                affected.append(ids[start + offset])

    return affected


def build_dense_run(
    data_dir: Path,
    qrels_path: Path,
    settings: RunConfig,
    model_path: Path,
    qdrant_url: str,
) -> dict[str, object]:
    """Encode the full corpus, index it, and rank selected queries."""
    input_paths = {
        "corpus": data_dir / "corpus.jsonl",
        "queries": data_dir / "queries.jsonl",
        "qrels": qrels_path,
    }
    inputs = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in input_paths.items()
    }

    dataset = load_dataset(data_dir, qrels_path)
    encoder, revision = load_encoder(model_path)

    document_ids = sorted(dataset.documents)
    texts = [
        (
            f"{dataset.documents[document_id].title}\n"
            f"{dataset.documents[document_id].text}"
            if dataset.documents[document_id].title
            else dataset.documents[document_id].text
        )
        for document_id in document_ids
    ]

    query_ids = sorted(dataset.queries)
    query_texts = [dataset.queries[query_id] for query_id in query_ids]

    encoder_metadata = {
        "model_id": MODEL_ID,
        "revision": revision,
        "dimension": DIMENSION,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "normalize_embeddings": True,
        "device": "cpu",
        "batch_size": 32,
        "document_text_policy": TEXT_POLICY,
        "implementation": "retrieval-lab-dense-v1",
        "versions": {
            name: version(name)
            for name in (
                "sentence-transformers",
                "transformers",
                "torch",
                "numpy",
                "qdrant-client",
            )
        },
    }

    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "corpus_sha256": inputs["corpus"]["sha256"],
                "encoder": encoder_metadata,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    collection_name = f"retrieval_lab_dense_{fingerprint}"

    truncated_documents = truncated_ids(encoder, document_ids, texts)
    truncated_queries = truncated_ids(encoder, query_ids, query_texts)

    vectors = encoder.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    results = {}

    with closing(QdrantClient(url=qdrant_url, timeout=60)) as client:
        server_version = client.info().version

        index = DenseIndex(
            client,
            collection_name,
            dimension=DIMENSION,
        )
        index.upsert(document_ids, vectors)

        for query_id in query_ids:
            query = dataset.queries[query_id]

            # Match the existing no-searchable-content convention
            # without applying BM25 tokenization to model inputs.
            if not any(character.isalnum() for character in query):
                hits = []
            else:
                query_vector = encoder.encode(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]
                hits = index.search(query_vector, settings.top_k)

            results[query_id] = [
                {"rank": rank, **hit} for rank, hit in enumerate(hits, start=1)
            ]

    # Detect ordinary input changes during this run.
    for name, path in input_paths.items():
        if sha256_file(path) != inputs[name]["sha256"]:
            raise ValueError(f"Input changed during retrieval: {name}")

    relevant_document_ids = {
        document_id
        for judgments in dataset.qrels.values()
        for document_id, grade in judgments.items()
        if grade > 0
    }
    affected_relevant = relevant_document_ids.intersection(truncated_documents)

    return {
        "schema_version": 1,
        "metadata": {
            "method": "dense",
            "config": settings.model_dump(),
            "encoder": encoder_metadata,
            "collection": collection_name,
            "index_fingerprint": fingerprint,
            "index_policy": "re-encode and upsert every run",
            "qdrant_server_version": server_version,
            "distance": "cosine",
            "exact_search": True,
            "tie_break": (
                "score descending, document_id ascending within returned candidate set"
            ),
            "empty_query_policy": ("no alphanumeric characters: return no hits"),
            "score_filter": "none",
            "documents": len(document_ids),
            "queries": len(query_ids),
            "inputs": inputs,
            "truncation": {
                "document_ids": truncated_documents,
                "query_ids": truncated_queries,
                "unique_relevant_documents": len(relevant_document_ids),
                "truncated_unique_relevant_documents": len(affected_relevant),
                "truncated_relevant_fraction": (
                    len(affected_relevant) / len(relevant_document_ids)
                ),
            },
        },
        "results": results,
    }

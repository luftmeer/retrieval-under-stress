import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from retrieval_lab.api.app import create_app
from retrieval_lab.api.service import ApiResources
from retrieval_lab.data.loader import Dataset, Document
from retrieval_lab.api.service import ApiSettings


class FakeSession:
    def __init__(self):
        self.calls = []

    def search_hits(
        self,
        method,
        query,
        *,
        top_k,
    ):
        self.calls.append(
            {
                "method": method,
                "query": query,
                "top_k": top_k,
            }
        )

        hits = [
            {
                "rank": 1,
                "document_id": "d1",
                "score": 0.9,
            },
            {
                "rank": 2,
                "document_id": "d2",
                "score": 0.7,
            },
        ]

        return hits[:top_k]


class FakeClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def api():
    session = FakeSession()
    client = FakeClient()

    dataset = Dataset(
        documents={
            "d1": Document(
                title="First",
                text="First document.",
            ),
            "d2": Document(
                title="Second",
                text="Second document.",
            ),
        },
        queries={},
        qrels={},
    )

    resources = ApiResources(
        dataset=dataset,
        session=session,
        client=client,
        model_revision="a" * 40,
        collection_name="test_collection",
        index_reused=True,
    )

    app = create_app(
        resource_factory=lambda settings: resources,
    )

    with TestClient(app) as http:
        yield http, session, client

    assert client.closed


def test_health_reports_loaded_resources(api):
    client, _, _ = api

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["documents"] == 2
    assert body["methods"] == [
        "bm25",
        "dense",
        "hybrid",
    ]
    assert body["index_reused"] is True


def test_search_returns_ranked_documents(api):
    client, session, _ = api

    response = client.post(
        "/search",
        json={
            "query": "  cellular response  ",
            "method": "hybrid",
            "top_k": 2,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["query"] == "cellular response"
    assert body["method"] == "hybrid"
    assert body["top_k"] == 2

    assert body["hits"] == [
        {
            "rank": 1,
            "document_id": "d1",
            "score": 0.9,
            "title": "First",
            "text": "First document.",
        },
        {
            "rank": 2,
            "document_id": "d2",
            "score": 0.7,
            "title": "Second",
            "text": "Second document.",
        },
    ]

    assert session.calls == [
        {
            "method": "hybrid",
            "query": "cellular response",
            "top_k": 2,
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "query": "",
            "method": "hybrid",
            "top_k": 10,
        },
        {
            "query": "   ",
            "method": "hybrid",
            "top_k": 10,
        },
        {
            "query": "claim",
            "method": "unknown",
            "top_k": 10,
        },
        {
            "query": "claim",
            "method": "dense",
            "top_k": 0,
        },
        {
            "query": "claim",
            "method": "dense",
            "top_k": 101,
        },
        {
            "query": "claim",
            "method": "dense",
            "top_k": 10,
            "unexpected": True,
        },
    ],
)
def test_invalid_requests_are_rejected(
    api,
    payload,
):
    client, _, _ = api

    response = client.post(
        "/search",
        json=payload,
    )

    assert response.status_code == 422



def test_api_settings_read_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "RETRIEVAL_DATA_DIR",
        "/data/clean",
    )
    monkeypatch.setenv(
        "RETRIEVAL_MODEL_PATH",
        "/app/configs/dense-model.json",
    )
    monkeypatch.setenv(
        "RETRIEVAL_QDRANT_URL",
        "http://qdrant:6333",
    )

    settings = ApiSettings.from_env()

    assert settings.data_dir == Path("/data/clean")

    assert settings.model_path == Path(
        "/app/configs/dense-model.json"
    )

    assert settings.qdrant_url == "http://qdrant:6333"

    assert settings.qrels_path == Path(
        "/data/clean/qrels.tsv"
    )
    
    
def test_api_settings_have_local_defaults(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "RETRIEVAL_DATA_DIR",
        raising=False,
    )
    monkeypatch.delenv(
        "RETRIEVAL_MODEL_PATH",
        raising=False,
    )
    monkeypatch.delenv(
        "RETRIEVAL_QDRANT_URL",
        raising=False,
    )

    settings = ApiSettings.from_env()

    assert settings.data_dir == Path(
        "data/processed/stress-test-v1/clean"
    )

    assert settings.qdrant_url == (
        "http://localhost:6333"
    )
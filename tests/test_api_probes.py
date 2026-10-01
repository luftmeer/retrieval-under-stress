from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from retrieval_lab.api.probes import router


def make_app(resources=None):
    app = FastAPI()
    app.include_router(router)

    if resources is not None:
        app.state.resources = resources

    return app


def make_resources(count=2):
    client = Mock()
    client.count.return_value = SimpleNamespace(count=count)

    return SimpleNamespace(
        client=client,
        collection_name="test-collection",
        dataset=SimpleNamespace(documents=["a", "b"]),
    )


def test_liveness_does_not_require_resources():
    with TestClient(make_app()) as client:
        assert client.get("/livez").status_code == 200


def test_readiness_requires_initialized_resources():
    with TestClient(make_app()) as client:
        assert client.get("/readyz").status_code == 503


@pytest.mark.parametrize(
    ("count", "expected_status"),
    [(2, 200), (1, 503), (0, 503)],
)
def test_readiness_checks_index_completeness(count, expected_status):
    resources = make_resources(count)

    with TestClient(make_app(resources)) as client:
        response = client.get("/readyz")

    assert response.status_code == expected_status

    resources.client.count.assert_called_once_with(
        collection_name="test-collection",
        exact=True,
        timeout=2,
    )


def test_qdrant_failure_changes_readiness_but_not_liveness():
    resources = make_resources()
    resources.client.count.side_effect = RuntimeError("Unavailable")

    with TestClient(make_app(resources)) as client:
        assert client.get("/readyz").status_code == 503
        assert client.get("/livez").status_code == 200

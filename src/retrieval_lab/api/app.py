"""FastAPI application exposing the retrieval system."""

from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from retrieval_lab.api.schemas import (
    HealthResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from retrieval_lab.api.service import (
    ApiResources,
    ApiSettings,
    build_resources,
)

from retrieval_lab.api.probes import router as probes_router

ResourceFactory = Callable[
    [ApiSettings],
    ApiResources,
]


def create_app(
    settings: ApiSettings | None = None,
    resource_factory: ResourceFactory = build_resources,
) -> FastAPI:
    settings = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resources = resource_factory(settings)
        app.state.resources = resources

        try:
            yield

        finally:
            resources.close()

    app = FastAPI(
        title="Retrieval Under Stress",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    app.include_router(probes_router)

    @app.get(
        "/health",
        response_model=HealthResponse,
    )
    def health(request: Request) -> HealthResponse:
        resources: ApiResources = request.app.state.resources

        return HealthResponse(
            status="ok",
            documents=len(resources.dataset.documents),
            methods=[
                "bm25",
                "dense",
                "hybrid",
            ],
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            model_revision=resources.model_revision,
            collection=resources.collection_name,
            index_reused=resources.index_reused,
        )

    @app.post(
        "/search",
        response_model=SearchResponse,
    )
    def search(
        payload: SearchRequest,
        request: Request,
    ) -> SearchResponse:
        resources: ApiResources = request.app.state.resources

        with resources.search_lock:
            raw_hits = resources.session.search_hits(
                payload.method,
                payload.query,
                top_k=payload.top_k,
            )

        hits = []

        for hit in raw_hits:
            document_id = hit["document_id"]

            document = resources.dataset.documents.get(document_id)

            if document is None:
                raise RuntimeError(f"Retriever returned unknown document {document_id!r}")

            hits.append(
                SearchHit(
                    rank=hit["rank"],
                    document_id=document_id,
                    score=hit["score"],
                    title=document.title,
                    text=document.text,
                )
            )

        return SearchResponse(
            query=payload.query,
            method=payload.method,
            top_k=payload.top_k,
            hits=hits,
        )

    return app


app = create_app()

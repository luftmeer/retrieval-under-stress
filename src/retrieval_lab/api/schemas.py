"""Validated HTTP request and response models."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

RetrievalMethod = Literal[
    "bm25",
    "dense",
    "hybrid",
]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(max_length=1000)
    method: RetrievalMethod = "hybrid"
    top_k: int = Field(default=10, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("query must not be blank")

        return value


class SearchHit(BaseModel):
    rank: int
    document_id: str
    score: float
    title: str
    text: str


class SearchResponse(BaseModel):
    query: str
    method: RetrievalMethod
    top_k: int
    hits: list[SearchHit]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    documents: int
    methods: list[RetrievalMethod]
    model_id: str
    model_revision: str
    collection: str
    index_reused: bool

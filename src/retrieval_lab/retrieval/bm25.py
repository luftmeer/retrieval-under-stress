"""BM25 retrieval with explicit preprocessing and deterministic ranking."""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from retrieval_lab.data.loader import Document

TOKEN_PATTERN = r"[^\W_]+"
TOKENIZER_NAME = "unicode-alphanumeric-casefold-v1"

BM25_PARAMETERS = {"k1": 1.5, "b": 0.75, "epsilon": 0.25}


def tokenize(text: str) -> list[str]:
    """Case-fold text and extract Unicode alphanumeric sequences."""
    return re.findall(TOKEN_PATTERN, text.casefold())


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    score: float


class BM25Retriever:
    """Build one in-memory index and reuse it across queries."""

    def __init__(self, documents: dict[str, Document]) -> None:
        if not documents:
            raise ValueError("Cannot index an empty corpus")

        self._document_ids = sorted(documents)
        tokenized_documents = []

        for document_id in self._document_ids:
            tokens = tokenize(documents[document_id].search_text)

            if not tokens:
                raise ValueError(f"Document {document_id!r} has no searchable tokens")

            tokenized_documents.append(tokens)

        self._index = BM25Okapi(tokenized_documents, **BM25_PARAMETERS)

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        """Return up to top_k hits, ordered by score and then document ID."""
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)

        hits = [
            SearchHit(document_id=document_id, score=float(score))
            for document_id, score in zip(
                self._document_ids,
                scores,
                strict=True,
            )
        ]

        hits.sort(key=lambda hit: (-hit.score, hit.document_id))

        return hits[:top_k]

"""One-query search using resources that have already been loaded."""

from retrieval_lab.retrieval.rrf import reciprocal_rank_fusion


class SearchSession:
    def __init__(
        self,
        bm25,
        encoder,
        index,
        *,
        candidate_depth: int = 100,
    ):
        if candidate_depth < 1:
            raise ValueError("candidate_depth must be positive")

        self.bm25 = bm25
        self.encoder = encoder
        self.index = index
        self.candidate_depth = candidate_depth

    def _bm25_hits(self, query: str) -> list[dict]:
        hits = self.bm25.search(
            query,
            top_k=self.candidate_depth,
        )

        return [
            {
                "rank": rank,
                "document_id": hit.document_id,
                "score": float(hit.score),
            }
            for rank, hit in enumerate(hits, start=1)
        ]

    def _dense_hits(self, query: str) -> list[dict]:
        if not any(character.isalnum() for character in query):
            return []

        vector = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        hits = self.index.search(
            vector,
            top_k=self.candidate_depth,
        )

        return [
            {
                "rank": rank,
                "document_id": hit["document_id"],
                "score": float(hit["score"]),
            }
            for rank, hit in enumerate(hits, start=1)
        ]

    def search_hits(
        self,
        method: str,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[dict]:
        if top_k is None:
            top_k = self.candidate_depth

        if type(top_k) is not int or top_k < 1 or top_k > self.candidate_depth:
            raise ValueError("top_k must be between 1 and candidate_depth")

        if method == "bm25":
            return self._bm25_hits(query)[:top_k]

        if method == "dense":
            return self._dense_hits(query)[:top_k]

        if method == "hybrid":
            lexical = self._bm25_hits(query)
            dense = self._dense_hits(query)

            return reciprocal_rank_fusion(
                [hit["document_id"] for hit in lexical],
                [hit["document_id"] for hit in dense],
                top_k=top_k,
                rrf_k=60,
            )

        raise ValueError(f"Unknown method: {method}")

    def search(
        self,
        method: str,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[str]:
        return [
            hit["document_id"]
            for hit in self.search_hits(
                method,
                query,
                top_k=top_k,
            )
        ]

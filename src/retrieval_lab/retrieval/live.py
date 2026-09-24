"""One-query search using resources that have already been loaded."""

from retrieval_lab.retrieval.rrf import reciprocal_rank_fusion


class SearchSession:
    def __init__(self, bm25, encoder, index):
        self.bm25 = bm25
        self.encoder = encoder
        self.index = index

    def search(self, method, query):
        if method == "bm25":
            return [hit.document_id for hit in self.bm25.search(query, top_k=100)]

        if method == "dense":
            if not any(character.isalnum() for character in query):
                return []

            vector = self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]

            return [hit["document_id"] for hit in self.index.search(vector, top_k=100)]

        if method == "hybrid":
            lexical = self.search("bm25", query)
            dense = self.search("dense", query)

            return [
                hit["document_id"]
                for hit in reciprocal_rank_fusion(
                    lexical,
                    dense,
                    top_k=100,
                    rrf_k=60,
                )
            ]

        raise ValueError(f"Unknown method: {method}")

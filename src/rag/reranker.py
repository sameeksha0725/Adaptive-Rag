"""
Cross-encoder reranking helper for the hybrid retrieval pipeline.

This module is intentionally separate so the reranking logic remains modular
and can be configured independently from the retriever interface.
"""

from sentence_transformers import CrossEncoder

from src.config.settings import Config


class CrossEncoderReranker:
    def __init__(self, config: Config):
        self.enabled = config.reranker_enabled()
        self.model_name = config.reranker_model()
        self.pool_size = config.reranker_pool_size()
        self.output_size = config.reranker_output_size()
        self.model = None

        if self.enabled:
            self.model = CrossEncoder(self.model_name)

    def rerank_documents(self, query: str, candidates: list[dict]) -> list[dict]:
        """Rerank candidate documents and return the top output_size results."""
        if not self.enabled or self.model is None:
            return candidates[: self.output_size]

        texts = [candidate["text"] for candidate in candidates]
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores), key=lambda item: item[1], reverse=True
        )

        # enrich returned candidates with reranker score and preserve meta
        out = []
        for candidate, score in ranked[: self.output_size]:
            c = dict(candidate)
            meta = dict(c.get("meta", {}) or {})
            meta["reranker_score"] = float(score)
            c["meta"] = meta
            out.append(c)

        return out

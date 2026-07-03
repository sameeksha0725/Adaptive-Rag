"""Cross-encoder reranking helper for the hybrid retrieval pipeline."""

from src.config.settings import Config


class CrossEncoderReranker:
    def __init__(self, config: Config):
        self.enabled = config.reranker_enabled()
        self.model_name = config.reranker_model()
        self.pool_size = config.reranker_pool_size()
        self.output_size = config.reranker_output_size()
        self.model = None
        self._load_attempted = False

    def _load_model(self) -> None:
        """Load the reranker model lazily on first use."""
        if self._load_attempted or not self.enabled:
            return

        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            self.enabled = False
            self.model = None
            print(f"[reranker] Unable to initialize reranker model {self.model_name}: {exc}")

    def rerank_documents(self, query: str, candidates: list[dict]) -> list[dict]:
        """Rerank candidate documents and return the top output_size results."""
        if not self.enabled:
            return candidates[: self.output_size]

        if self.model is None:
            self._load_model()

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

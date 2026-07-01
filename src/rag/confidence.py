"""Modular confidence scoring for retrieved / reranked candidates.

This module provides a `ConfidenceScorer` class that computes a
normalized confidence percentage (0-100) given reranked candidates.

Candidates are expected to be a list of dicts with a `meta` mapping
that may contain any of: `retrieval_score` (FAISS/BM25),
`rrf_score`, and `reranker_score`.

The scoring logic is intentionally simple and replaceable: it normalizes
component scores per-query and then computes a weighted sum. Weights
and normalization behavior can be changed without touching callers.
"""

from typing import List


class ConfidenceScorer:
    def __init__(
        self,
        weight_semantic: float = 0.25,
        weight_bm25: float = 0.15,
        weight_rrf: float = 0.30,
        weight_reranker: float = 0.30,
    ):
        # weights should sum to 1.0 but we will normalize them defensively
        total = weight_semantic + weight_bm25 + weight_rrf + weight_reranker
        self.w_sem = weight_semantic / total
        self.w_bm = weight_bm25 / total
        self.w_rrf = weight_rrf / total
        self.w_rer = weight_reranker / total

    def _safe_minmax_norm(self, values: List[float]):
        """Min-max normalize a list of values to [0,1]. Handles constant lists."""
        if not values:
            return []
        minv = min(values)
        maxv = max(values)
        if maxv - minv <= 1e-12:
            return [0.5 for _ in values]
        return [(v - minv) / (maxv - minv) for v in values]

    def score(self, candidates: List[dict]) -> float:
        """Compute overall confidence (0-100) for the provided candidates.

        The method extracts component scores from candidate['meta'].
        Supported keys in meta: `retrieval_score` (semantic or bm25),
        `rrf_score`, `reranker_score`.
        """
        # collect raw component lists
        semantic_vals = []
        bm25_vals = []
        rrf_vals = []
        rerank_vals = []

        for c in candidates:
            meta = c.get("meta", {}) or {}
            # heuristics: if source indicates BM25, treat retrieval_score as bm25
            method = meta.get("retrieval_method", "") or ""
            val = meta.get("retrieval_score")
            if val is None:
                # treat missing as 0 for normalization purposes
                val = 0.0

            if method.upper().startswith("BM") or "BM25" in method.upper():
                bm25_vals.append(float(val))
            else:
                semantic_vals.append(float(val))

            rrf_vals.append(float(meta.get("rrf_score") or 0.0))
            rerank_vals.append(float(meta.get("reranker_score") or 0.0))

        # normalize components individually
        sem_norm = self._safe_minmax_norm(semantic_vals) if semantic_vals else []
        bm_norm = self._safe_minmax_norm(bm25_vals) if bm25_vals else []
        rrf_norm = self._safe_minmax_norm(rrf_vals) if rrf_vals else []
        rer_norm = self._safe_minmax_norm(rerank_vals) if rerank_vals else []

        # For aggregation, average per-component normalized values (if present)
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        sem_score = avg(sem_norm)
        bm_score = avg(bm_norm)
        rrf_score = avg(rrf_norm)
        rer_score = avg(rer_norm)

        # Weighted combination
        combined = (
            self.w_sem * sem_score
            + self.w_bm * bm_score
            + self.w_rrf * rrf_score
            + self.w_rer * rer_score
        )

        # map to percentage 0-100
        percent = max(0.0, min(100.0, combined * 100.0))
        return round(percent)

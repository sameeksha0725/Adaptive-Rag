"""Hybrid retriever that combines FAISS and BM25 results using
Reciprocal Rank Fusion (RRF) and an optional Cross-Encoder reranking
stage.

This retriever preserves the existing retriever interface but exposes
configuration to control the number of candidates retrieved before
reranking and the final number of documents returned to the caller.
"""

from typing import List, Optional

from langchain_core.documents import Document


class HybridRetriever:
    def __init__(
        self,
        semantic_retriever,
        keyword_retriever,
        k: int = 4,
        rrf_k: int = 60,
        reranker: Optional[object] = None,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
    ):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.k = k
        self.rrf_k = rrf_k
        self.reranker = reranker
        # If provided, these control the end-to-end pipeline sizes.
        self.retrieval_top_k = retrieval_top_k or k
        self.final_top_k = final_top_k or k

    def _document_key(self, document: Document) -> str:
        return (
            getattr(document, "page_content", None)
            or getattr(document, "content", None)
            or str(document)
        )

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """Return a list of Documents for the query.

        Workflow:
        - retrieve up to `retrieval_top_k` from semantic and keyword retrievers
        - fuse results using RRF
        - optionally rerank fused candidates using a Cross-Encoder
        - return the top `final_top_k` Documents
        """
        retrieval_k = self.retrieval_top_k
        final_k = self.final_top_k

        # get candidate lists
        semantic_docs = self.semantic_retriever.retrieve(query, k=retrieval_k)
        keyword_docs = self.keyword_retriever.retrieve(query, k=retrieval_k)

        scores = {}
        doc_by_key = {}

        def add_list(doc_list):
            for rank, doc in enumerate(doc_list, start=1):
                key = self._document_key(doc)
                doc_by_key.setdefault(key, doc)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)

        add_list(semantic_docs)
        add_list(keyword_docs)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_keys = [key for key, _score in ranked][:retrieval_k]

        # update document metadata to indicate hybrid retrieval and attach the
        # aggregated RRF score so downstream components (reranker, generator)
        # can display citations and scores.
        for key, score in ranked:
            if key in doc_by_key:
                doc = doc_by_key[key]
                meta = dict(getattr(doc, "metadata", {}) or {})
                meta["retrieval_method"] = "Hybrid"
                meta["retrieval_score"] = float(score)
                doc.metadata = meta

        # prepare candidate dicts for reranker if present
        candidates = [
            {"text": key, "meta": getattr(doc_by_key[key], "metadata", {})}
            for key in top_keys
        ]

        if self.reranker is not None and getattr(self.reranker, "enabled", False):
            reranked = self.reranker.rerank_documents(query, candidates)
            # reranker returns dicts with 'text'; map back to Document objects
            final_texts = [c["text"] for c in reranked][:final_k]
        else:
            final_texts = top_keys[:final_k]

        return [doc_by_key[text] for text in final_texts if text in doc_by_key]

    def get_relevant_documents(self, query: str, k: int = None):
        return self.retrieve(query, k=k)

    def invoke(self, query: str):
        docs = self.retrieve(query, k=self.k)
        return "\n\n".join(
            getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
            for d in docs
        )

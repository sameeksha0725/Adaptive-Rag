"""
BM25 keyword retriever implementation using rank_bm25.

This module exposes the same retriever interface as the rest of the
application while using rank_bm25 internally.
"""

import re
from typing import List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str):
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.corpus = [
            tokenize(getattr(doc, "page_content", None) or getattr(doc, "content", None) or "")
            for doc in documents
        ]
        self.bm25 = BM25Okapi(self.corpus)

    @classmethod
    def from_documents(cls, documents: List[Document]):
        return cls(documents)

    def retrieve(self, query: str, k: int = 4):
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda pair: pair[1], reverse=True
        )[:k]

        results = []
        for index, score in ranked:
            doc = self.documents[index]
            # preserve metadata and add retrieval info
            meta = dict(getattr(doc, "metadata", {}) or {})
            meta["retrieval_method"] = "BM25"
            meta["retrieval_score"] = float(score)
            doc.metadata = meta
            results.append(doc)

        return results

    def get_relevant_documents(self, query: str, k: int = 4):
        return self.retrieve(query, k=k)

    def invoke(self, query: str):
        docs = self.retrieve(query, k=4)
        return "\n\n".join(
            getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
            for d in docs
        )

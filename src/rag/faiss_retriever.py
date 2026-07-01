"""
FAISS semantic retriever implementation.

This module exposes a simple retriever class with the same interface the
rest of the application expects: `invoke(query)` and `retrieve(query, k)`.
"""

from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.llms.embeddings import embeddings


class FAISSRetriever:
    def __init__(self, vectorstore: FAISS, documents: List[Document]):
        self.vectorstore = vectorstore
        self.documents = documents

    @classmethod
    def from_documents(cls, documents: List[Document]):
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
        return cls(vectorstore=vectorstore, documents=documents)

    def retrieve(self, query: str, k: int = 4):
        if self.vectorstore is None:
            return []

        # Prefer methods that return scores when available
        if hasattr(self.vectorstore, "similarity_search_with_score"):
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            results = []
            for doc, score in docs_and_scores:
                meta = dict(getattr(doc, "metadata", {}) or {})
                meta["retrieval_method"] = "FAISS"
                meta["retrieval_score"] = float(score)
                doc.metadata = meta
                results.append(doc)
            return results

        if hasattr(self.vectorstore, "similarity_search"):
            docs = self.vectorstore.similarity_search(query, k=k)
            # attach retrieval metadata, score unknown
            for doc in docs:
                meta = dict(getattr(doc, "metadata", {}) or {})
                meta["retrieval_method"] = "FAISS"
                meta["retrieval_score"] = None
                doc.metadata = meta
            return docs

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
            for doc in docs:
                meta = dict(getattr(doc, "metadata", {}) or {})
                meta["retrieval_method"] = "FAISS"
                meta["retrieval_score"] = None
                doc.metadata = meta
            return docs
        if hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(query)
            for doc in docs:
                meta = dict(getattr(doc, "metadata", {}) or {})
                meta["retrieval_method"] = "FAISS"
                meta["retrieval_score"] = None
                doc.metadata = meta
            return docs
        if hasattr(retriever, "get_relevant_entries"):
            entries = retriever.get_relevant_entries(query)
            # entries may be different shape; attempt to return documents
            return entries

        return []

    def get_relevant_documents(self, query: str, k: int = 4):
        return self.retrieve(query, k=k)

    def invoke(self, query: str):
        docs = self.retrieve(query, k=4)
        return "\n\n".join(
            getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
            for d in docs
        )

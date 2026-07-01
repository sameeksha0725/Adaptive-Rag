"""
Embeddings shim: tries to use OpenAIEmbeddings if available, otherwise
provides a clear runtime error explaining embeddings need configuration.

This keeps the rest of the codebase unchanged while removing a hard
dependency on OpenAI at import time.
"""
from typing import List

try:
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
except Exception:
    class _EmbeddingsUnavailable:
        def embed_documents(self, documents: List[str]):
            raise RuntimeError(
                "OpenAI embeddings are not available. Install and configure an embeddings provider or set OPENAI_API_KEY."
            )

        def embed_query(self, query: str):
            raise RuntimeError(
                "OpenAI embeddings are not available. Install and configure an embeddings provider or set OPENAI_API_KEY."
            )

    embeddings = _EmbeddingsUnavailable()

"""Embedding provider wrapper using HuggingFace embeddings."""

import os
from typing import List

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
except Exception:
    class _EmbeddingsUnavailable:
        def embed_documents(self, documents: List[str]):
            raise RuntimeError(
                "Embeddings are not available. Install the required HuggingFace dependencies or configure EMBEDDING_MODEL."
            )

        def embed_query(self, query: str):
            raise RuntimeError(
                "Embeddings are not available. Install the required HuggingFace dependencies or configure EMBEDDING_MODEL."
            )

    embeddings = _EmbeddingsUnavailable()

"""
Retriever setup and vector store configuration.
"""

import os

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool

from src.rag.faiss_retriever import FAISSRetriever
from src.rag.bm25_retriever import BM25Retriever
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import CrossEncoderReranker
from src.config.settings import Config
from src.llms.embeddings import embeddings

_hybrid_retriever = None
_documents = None


def _build_hybrid_retriever(documents: list[Document]):
    config = Config()
    semantic_retriever = FAISSRetriever.from_documents(documents)
    keyword_retriever = BM25Retriever.from_documents(documents)
    reranker = CrossEncoderReranker(config)
    return HybridRetriever(
        semantic_retriever=semantic_retriever,
        keyword_retriever=keyword_retriever,
        k=config.final_top_k(),
        rrf_k=60,
        reranker=reranker,
        retrieval_top_k=config.retrieval_top_k(),
        final_top_k=config.final_top_k(),
    )


def _create_retriever_tool(retriever, description: str = None):
    instruction = (
        f"Use this tool **only** to answer questions about: {description}\n"
        "Don't use this tool to answer anything else."
    )
    return create_retriever_tool(
        retriever,
        "retriever_customer_uploaded_documents",
        instruction,
    )


def retriever_chain(chunks: list[Document]):
    """
    Initialize and store the hybrid retriever with uploaded documents.

    Args:
        chunks: List of document chunks to store.

    Returns:
        Boolean indicating success of the operation.
    """
    global _hybrid_retriever, _documents

    try:
        _documents = chunks
        _hybrid_retriever = _build_hybrid_retriever(chunks)

        print("Hybrid retriever initialized with documents")
        print(f"Retriever contains {len(chunks)} document chunks")
        return True
    except Exception as e:
        print(f"Error initializing hybrid retriever: {e}")
        return False


def _get_description():
    if os.path.exists("description.txt"):
        with open("description.txt", "r", encoding="utf-8") as f:
            return f.read()
    return None


def _ensure_dummy_retriever():
    global _hybrid_retriever, _documents
    if _hybrid_retriever is not None:
        return

    from langchain_core.documents import Document as LangChainDocument

    dummy_doc = LangChainDocument(
        page_content="No documents have been uploaded yet. Please upload a document first.",
        metadata={"source": "initialization"}
    )
    _documents = [dummy_doc]
    _hybrid_retriever = _build_hybrid_retriever(_documents)


def get_retriever():
    """
    Get the hybrid retriever tool configured for the stored document chunks.

    Returns:
        A retriever tool with the same API as the existing retriever.
    """
    _ensure_dummy_retriever()

    description = _get_description()
    return _create_retriever_tool(_hybrid_retriever, description)


def search_faiss_documents(query: str, k: int = 4):
    """Helper to search FAISS and return Documents with metadata.

    This is used by graph nodes that perform hybrid/reranking flows.
    """
    _ensure_dummy_retriever()
    semantic = FAISSRetriever.from_documents(_documents)

    # Prefer a method that returns scores
    if hasattr(semantic.vectorstore, "similarity_search_with_score"):
        docs_and_scores = semantic.vectorstore.similarity_search_with_score(query, k=k)
        results = []
        for doc, score in docs_and_scores:
            meta = dict(getattr(doc, "metadata", {}) or {})
            meta["retrieval_method"] = "FAISS"
            meta["retrieval_score"] = float(score)
            doc.metadata = meta
            results.append(doc)
        return results

    # fallback to standard retrieve which attaches metadata where possible
    return semantic.retrieve(query, k=k)


def search_bm25_documents(query: str, k: int = 4):
    _ensure_dummy_retriever()
    bm = BM25Retriever.from_documents(_documents)
    return bm.retrieve(query, k=k)

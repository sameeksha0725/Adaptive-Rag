"""Core configuration and environment settings."""

import os

from dotenv import load_dotenv

from src.config.settings import (
    EMBEDDING_MODEL,
    FINAL_TOP_K,
    OLLAMA_MODEL,
    RETRIEVAL_TOP_K,
    RERANKER_MODEL,
    RRF_K,
    USE_RERANKER,
)

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    CODE_COLLECTION = os.getenv("QDRANT_CODE_COLLECTION", "codebase")
    DOCS_COLLECTION = os.getenv("QDRANT_DOCS_COLLECTION", "guidelines")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", RERANKER_MODEL)
    USE_RERANKER: bool = os.getenv("USE_RERANKER", str(USE_RERANKER)).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", str(RETRIEVAL_TOP_K)))
    FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", str(FINAL_TOP_K)))
    RRF_K: int = int(os.getenv("RRF_K", str(RRF_K)))


settings = Settings()

os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
os.environ["OLLAMA_MODEL"] = settings.OLLAMA_MODEL
os.environ["EMBEDDING_MODEL"] = settings.EMBEDDING_MODEL
os.environ["RERANKER_MODEL"] = settings.RERANKER_MODEL
os.environ["USE_RERANKER"] = str(settings.USE_RERANKER).lower()
os.environ["RETRIEVAL_TOP_K"] = str(settings.RETRIEVAL_TOP_K)
os.environ["FINAL_TOP_K"] = str(settings.FINAL_TOP_K)
os.environ["RRF_K"] = str(settings.RRF_K)

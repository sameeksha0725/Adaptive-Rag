"""
Core configuration and environment settings.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    CODE_COLLECTION = os.getenv("QDRANT_CODE_COLLECTION", "codebase")
    DOCS_COLLECTION = os.getenv("QDRANT_DOCS_COLLECTION", "guidelines")
    # Ollama model to use for local LLM inference
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:latest")


settings = Settings()

# Set env variables for LangChain integrations
# Set env variables for LangChain/Tavily integrations
os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
# Expose ollama model name to environment (optional)
os.environ["OLLAMA_MODEL"] = settings.OLLAMA_MODEL

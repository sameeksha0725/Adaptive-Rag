"""Runtime configuration helpers for the Adaptive RAG application."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Load and manage configuration from YAML file and environment variables."""

    def __init__(self, config_file: str | None = None):
        """Initialize configuration from a YAML file."""
        base_path = Path(__file__).parent
        config_path = (
            base_path / "prompts.yaml" if config_file is None else Path(config_file)
        )
        with open(config_path, "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle) or {}

    def prompt(self, key: str) -> str:
        """Retrieve a prompt from configuration."""
        return self.config["prompts"][key]

    def _env_bool(self, env_key: str, default: bool) -> bool:
        """Read a boolean flag from the environment."""
        value = os.getenv(env_key)
        if value is None:
            return default
        return value.lower() in ("1", "true", "yes", "on")

    def _env_int(self, env_key: str, default: int) -> int:
        """Read an integer value from the environment."""
        value = os.getenv(env_key)
        return int(value) if value is not None else default

    def ollama_model(self) -> str:
        """Return the configured Ollama model name."""
        return os.getenv("OLLAMA_MODEL", "qwen3:latest")

    def embedding_model(self) -> str:
        """Return the configured embedding model name."""
        return os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    def reranker_enabled(self) -> bool:
        """Return whether cross-encoder reranking is enabled."""
        default = bool(self.config.get("reranker", {}).get("use_reranker", True))
        return self._env_bool("USE_RERANKER", default)

    def reranker_model(self) -> str:
        """Return the configured reranker model name."""
        return os.getenv(
            "RERANKER_MODEL",
            self.config.get("reranker", {}).get("model", "BAAI/bge-reranker-base"),
        )

    def retrieval_top_k(self) -> int:
        """Return the number of candidate documents to retrieve before reranking."""
        default_value = int(self.config.get("reranker", {}).get("retrieval_top_k", 20))
        return self._env_int("RETRIEVAL_TOP_K", default_value)

    def final_top_k(self) -> int:
        """Return the number of documents to return after reranking."""
        default_value = int(self.config.get("reranker", {}).get("final_top_k", 5))
        return self._env_int("FINAL_TOP_K", default_value)

    def rrf_k(self) -> int:
        """Return the Reciprocal Rank Fusion parameter."""
        default_value = int(self.config.get("retriever", {}).get("rrf_k", 60))
        return self._env_int("RRF_K", default_value)

    def reranker_pool_size(self) -> int:
        """Backward compatible alias for retrieval_top_k."""
        return self.retrieval_top_k()

    def reranker_output_size(self) -> int:
        """Backward compatible alias for final_top_k."""
        return self.final_top_k()

    def retry_max(self) -> int:
        """Return the maximum number of retrieval retry attempts."""
        return int(self.config.get("retry", {}).get("max_retries", 2))

    def adaptive_max_retries(self) -> int:
        """Return the maximum number of rewrite retries for poor grades."""
        return int(self.config.get("adaptive", {}).get("max_retries", 2))


_default_config = Config()

OLLAMA_MODEL = _default_config.ollama_model()
EMBEDDING_MODEL = _default_config.embedding_model()
RERANKER_MODEL = _default_config.reranker_model()
USE_RERANKER = _default_config.reranker_enabled()
RETRIEVAL_TOP_K = _default_config.retrieval_top_k()
FINAL_TOP_K = _default_config.final_top_k()
RRF_K = _default_config.rrf_k()
MAX_RETRIES = _default_config.retry_max()
LLM_MODEL = OLLAMA_MODEL

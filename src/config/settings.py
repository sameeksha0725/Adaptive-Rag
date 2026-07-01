"""
Configuration settings for the application.
"""

import os
from pathlib import Path

import yaml


class Config:
    """Load and manage configuration from YAML file."""

    def __init__(self, config_file: str = None):
        """
        Initialize configuration from YAML file.

        Args:
            config_file: Optional path to config file. Defaults to prompts.yaml.
        """
        base_path = Path(__file__).parent
        config_path = (
            base_path / "prompts.yaml"
            if config_file is None
            else Path(config_file)
        )
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def prompt(self, key: str) -> str:
        """
        Retrieve a prompt from configuration.

        Args:
            key: The prompt key.

        Returns:
            The prompt template string.
        """
        return self.config["prompts"][key]

    def reranker_enabled(self) -> bool:
        """Return whether cross-encoder reranking is enabled."""
        return bool(self.config.get("reranker", {}).get("use_reranker", True))

    def reranker_model(self) -> str:
        """Return the configured reranker model name."""
        return self.config.get("reranker", {}).get("model", "BAAI/bge-reranker-base")

    def retrieval_top_k(self) -> int:
        """Return the number of candidate documents to retrieve before reranking."""
        return int(self.config.get("reranker", {}).get("retrieval_top_k", 20))

    def final_top_k(self) -> int:
        """Return the number of documents to return after reranking."""
        return int(self.config.get("reranker", {}).get("final_top_k", 5))

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


# Convenience module-level constants for quick access and for compatibility
# with code that expects simple settings variables.
_default_config = Config()
RERANKER_MODEL = _default_config.reranker_model()
USE_RERANKER = _default_config.reranker_enabled()
RETRIEVAL_TOP_K = _default_config.retrieval_top_k()
FINAL_TOP_K = _default_config.final_top_k()

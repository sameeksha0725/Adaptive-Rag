"""Compatibility wrapper that exposes the Ollama-backed llm for existing imports."""

from dotenv import load_dotenv

load_dotenv()

from src.llms.ollama import llm

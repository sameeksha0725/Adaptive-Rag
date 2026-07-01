"""
OpenAI LLM initialization and configuration.
"""
"""
Shim module kept for compatibility: exposes `llm` as before.

This file previously created a ChatOpenAI instance. It now imports the
Ollama adapter so other modules don't need to change their imports.
"""

from dotenv import load_dotenv

load_dotenv()

# Import the replacement Ollama `llm` so existing imports continue to work.
from src.llms.ollama import llm
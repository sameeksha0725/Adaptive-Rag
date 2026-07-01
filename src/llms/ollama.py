"""
Ollama LLM adapter to replace ChatOpenAI.

This module exposes `llm` to match the previous `openai.llm` variable so
other modules can continue importing `from src.llms.openai import llm`.

It prefers LangChain's `Ollama` LLM if available, and falls back to the
`ollama` python client. The model name is taken from `src.core.config.settings`.
"""
import os
from dotenv import load_dotenv

load_dotenv()

from src.core.config import settings

MODEL_NAME = getattr(settings, "OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:latest"))


try:
    # Prefer LangChain's Ollama wrapper if available
    from langchain.llms import Ollama as LangchainOllama

    llm = LangchainOllama(model=MODEL_NAME)
except Exception:
    # Fallback to the official ollama python client if langchain wrapper not present
    try:
        from ollama import Ollama as OllamaClient

        client = OllamaClient()

        class OllamaFallback:
            def __init__(self, model: str):
                self.model = model

            def __call__(self, prompt: str, **kwargs):
                resp = client.generate(self.model, prompt)
                if isinstance(resp, dict):
                    return resp.get("text") or resp.get("output") or str(resp)
                return str(resp)

            def invoke(self, input_data):
                # Accept list of message objects or dict input used by the codebase
                if isinstance(input_data, list):
                    prompt = "\n".join(getattr(m, "content", str(m)) for m in input_data)
                elif isinstance(input_data, dict):
                    # Use the first value as prompt
                    prompt = next(iter(input_data.values()))
                else:
                    prompt = str(input_data)

                resp = client.generate(self.model, prompt)

                class Result:
                    def __init__(self, content):
                        self.content = content

                if isinstance(resp, dict):
                    text = resp.get("text") or resp.get("output") or str(resp)
                else:
                    text = str(resp)

                return Result(text)

            def with_structured_output(self, model_class):
                # Best-effort: return self. Proper structured parsing requires
                # integration with LangChain/langgraph which is out of scope here.
                return self

        llm = OllamaFallback(MODEL_NAME)
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("Could not initialize Ollama LLM client") from e

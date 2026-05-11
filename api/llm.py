"""LLM client — thin wrapper around `langchain-openai`.

The endpoint, model, and api_key all come from `.env` (see `api.config`).
Default target is Google Gemini via its OpenAI-compatible bridge, but any
OpenAI-compatible endpoint (GitHub Models, Groq, Ollama, OpenRouter, internal
proxy) works with no code changes — just swap `LLM_BASE_URL`, `MODEL_NAME`,
`LLM_API_KEY` in `.env`.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import settings


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Chat model used by extract + render nodes.

    Lower temperature (≈ 0) is preferred for extraction (deterministic JSON
    output) and a moderate temperature (≈ 0.4) for render (natural phrasing).
    """
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature,
        timeout=30,
        max_retries=2,
    )


def shutdown_llm() -> None:
    """No-op — kept so callers (e.g. the FastAPI lifespan and smoke_test) can
    invoke it without changes. ChatOpenAI holds no long-lived resources that
    need explicit cleanup.
    """
    return None

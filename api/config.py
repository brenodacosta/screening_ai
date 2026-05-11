from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    # OpenAI-compatible endpoint. Default points at Google's Gemini bridge;
    # swap to any other compatible provider (GitHub Models, Groq, Ollama,
    # OpenRouter, internal proxy) by changing these three env vars.
    llm_api_key: str = "replace-me"
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # Model id as exposed by the endpoint. Examples:
    #   Gemini        : gemini-2.0-flash, gemini-2.0-flash-lite, gemini-2.5-flash
    #   GitHub Models : openai/gpt-4o-mini, openai/gpt-4o, ...
    #   Groq          : llama-3.3-70b-versatile, llama-3.1-8b-instant
    model_name: str = "gemini-2.0-flash"

    # --- Paths ---
    data_dir: Path = Path("./data")
    jd_path: Path = Path("./jobs/delivery_driver.md")
    checkpoint_db_path: Path = Path("./data/checkpoints.db")

    # --- Behavior ---
    inactivity_timeout_seconds: int = 600
    inactivity_sweep_interval_seconds: int = 60
    default_language: Literal["es", "en"] = "es"

    # --- Inter-service ---
    api_base_url: str = "http://localhost:8000"

    # --- Optional tracing ---
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "screening-ai"

    # --- App metadata ---
    agent_version: str = "0.1.0"
    job_id: str = "delivery_driver"


settings = Settings()

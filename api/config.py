from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    # OpenAI-compatible endpoint. Default points at GitHub Models; can be swapped
    # to any other provider (Gemini OpenAI-compat, Groq, Ollama, OpenRouter, ...).
    # Auth: the PAT (or provider API key) is passed as `api_key`.
    github_pat: str = "replace-me"
    llm_base_url: str = "https://models.github.ai/inference"
    # Model id as exposed by the endpoint. For GitHub Models use the prefixed
    # form: openai/gpt-4o-mini, openai/gpt-4o, meta/meta-llama-3.1-70b-instruct, ...
    model_name: str = "openai/gpt-4o-mini"

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

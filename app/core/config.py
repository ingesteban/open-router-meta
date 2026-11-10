"""Settings loaded from environment variables / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "meta/muse-spark-1.3"
    # Optional attribution headers shown on openrouter.ai rankings.
    app_url: str = "http://localhost:8000"
    app_title: str = "Muse Build Day Starter"
    request_timeout_seconds: float = 120.0
    # Debate Arena
    arena_concurrency: int = 30
    arena_reasoning_effort: str = "low"  # empty = don't send the reasoning param
    arena_max_agents: int = 2000


@lru_cache
def get_settings() -> Settings:
    return Settings()

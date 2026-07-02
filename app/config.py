"""Environment-driven configuration.

Only one required secret: GEMINI_API_KEY. Everything else has a sane default
so the service boots locally with just a .env file and on Render with just
one env var set in the dashboard.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2
    max_turns: int = 8
    catalog_path: str = "data/catalog.json"
    request_timeout_seconds: int = 25  # leaves headroom under the grader's 30s cap


@lru_cache
def get_settings() -> Settings:
    return Settings()

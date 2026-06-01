from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the chat app, read from the environment / .env.

    Field names map case-insensitively to env vars (lancedb_path <- LANCEDB_PATH).
    Required fields raise a clear ValidationError on first access rather than a bare
    KeyError at import time, which keeps the module importable in tests.
    """

    model_config = SettingsConfigDict(extra="ignore")

    lancedb_path: str
    ollama_host: str
    embed_model: str
    top_k: int = 15
    score_threshold: float = 0.0


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings()

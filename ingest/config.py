from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the ingestion run, read from the environment / .env.

    Zotero paths are resolved separately in zotero_reader (they have sensible defaults);
    these are the destinations and embedding endpoint the orchestrator needs.
    """

    model_config = SettingsConfigDict(extra="ignore")

    lancedb_path: str
    ollama_host: str
    embed_model: str


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings()

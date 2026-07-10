from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the ingestion run, read from the environment / .env.

    Zotero paths are resolved separately in zotero_reader (they have sensible defaults);
    embedding provider/model/dimension are owned by the embeddings package (read from
    EMBED_* env vars), so the orchestrator only needs the destination here.
    """

    model_config = SettingsConfigDict(extra="ignore")

    lancedb_path: str


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings()

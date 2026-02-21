from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_chat_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_CHAT_MODEL")
    openai_embed_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBED_MODEL")
    default_top_k: int = Field(default=8, alias="DEFAULT_TOP_K")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def default_index_dir(target_path: str | Path) -> Path:
    return Path(target_path).resolve() / ".mana_index"

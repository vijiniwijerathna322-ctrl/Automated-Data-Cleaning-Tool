"""12-factor configuration: every value can be overridden by a DATACLEANER_* env var."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATACLEANER_", env_file=".env", extra="ignore", frozen=True
    )

    max_upload_mb: int = Field(25, ge=1, le=1024)
    max_rows: int = Field(500_000, ge=1)
    log_level: str = "INFO"
    json_logs: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()

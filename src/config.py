"""Application settings loaded from environment variables / .env file.

All settings are typed and validated by Pydantic. Override any field via an
APP_-prefixed environment variable (e.g. APP_MAX_LEN=128).

The single canonical accessor is `get_settings()`, which is cached for the
lifetime of the process via `functools.lru_cache`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, sourced from env vars and `.env`.

    Precedence (highest first):
    1. Explicit kwargs to `Settings(...)` (used in tests).
    2. Environment variables prefixed `APP_`.
    3. Variables in a `.env` file at the project root.
    4. Field defaults below.
    """

    # ----- Paths -----
    model_checkpoint_path: Path = Path("artifacts/model")
    routing_config_path: Path = Path("configs/routing.yaml")
    db_url: str = "sqlite:///artifacts/tickets.db"

    # ----- Model and inference -----
    base_model_name: str = "bert-base-uncased"
    max_len: int = 64
    top_k: int = 3
    device: Literal["cuda", "cpu", "auto"] = "auto"

    # ----- Training (used by train.py only) -----
    batch_size: int = 64
    learning_rate: float = 2e-5
    num_epochs: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42

    # ----- Server -----
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    model_version: str = "bert-base-uncased@v0.1.0-dev"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process-wide Settings instance."""
    return Settings()

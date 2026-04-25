"""Tests for src.config (Settings, get_settings)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import Settings, get_settings


class TestSettingsDefaults:
    """Settings should provide architecture-defined defaults when no env vars are set."""

    def test_paths_have_expected_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear all APP_ env vars to ensure pure-default behavior
        for key in list(os.environ):
            if key.startswith("APP_"):
                monkeypatch.delenv(key, raising=False)

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.model_checkpoint_path == Path("artifacts/model")
        assert settings.routing_config_path == Path("configs/routing.yaml")
        assert settings.db_url == "sqlite:///artifacts/tickets.db"

    def test_model_and_inference_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in list(os.environ):
            if key.startswith("APP_"):
                monkeypatch.delenv(key, raising=False)

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.base_model_name == "bert-base-uncased"
        assert settings.max_len == 64
        assert settings.top_k == 3
        assert settings.device == "auto"

    def test_training_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in list(os.environ):
            if key.startswith("APP_"):
                monkeypatch.delenv(key, raising=False)

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.batch_size == 64
        assert settings.learning_rate == 2e-5
        assert settings.num_epochs == 4
        assert settings.weight_decay == 0.01
        assert settings.warmup_ratio == 0.1
        assert settings.seed == 42

    def test_server_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in list(os.environ):
            if key.startswith("APP_"):
                monkeypatch.delenv(key, raising=False)

        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.log_level == "INFO"
        assert settings.model_version == "bert-base-uncased@v0.1.0-dev"


class TestSettingsEnvOverride:
    """APP_-prefixed env vars override defaults."""

    def test_max_len_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_MAX_LEN", "128")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.max_len == 128

    def test_db_url_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_DB_URL", "sqlite:///:memory:")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.db_url == "sqlite:///:memory:"

    def test_log_level_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_LOG_LEVEL", "DEBUG")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.log_level == "DEBUG"

    def test_invalid_log_level_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_LOG_LEVEL", "VERBOSE")  # not in Literal
        with pytest.raises(ValueError):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_invalid_device_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_DEVICE", "tpu")  # not in Literal
        with pytest.raises(ValueError):
            Settings(_env_file=None)  # type: ignore[call-arg]


class TestGetSettings:
    """get_settings is cached and returns the same instance per process."""

    def test_returns_settings_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_is_cached(self) -> None:
        a = get_settings()
        b = get_settings()
        assert a is b

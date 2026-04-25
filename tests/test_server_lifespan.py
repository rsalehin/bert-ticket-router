"""Tests for api/server.py Ã¢â‚¬â€ FastAPI app construction and lifespan (T-021)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from transformers import PreTrainedTokenizerBase

from src.errors import ModelNotLoadedError
from src.model import BertClassifier

_NUM_LABELS = 10
_FAKE_LABELS = [f"intent_{i:02d}" for i in range(_NUM_LABELS)]
_DEPARTMENTS = ["Cards", "Transfers", "Account", "Security", "General"]


@pytest.fixture(scope="module")
def tiny_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
    tiny_bert_name: str,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> Path:
    ckpt = tmp_path_factory.mktemp("ckpt")
    model = BertClassifier(base_model_name=tiny_bert_name, num_labels=_NUM_LABELS)
    manifest = {
        "model_version": "bert-tiny@test",
        "base_model": tiny_bert_name,
        "trained_at": "2026-01-01T00:00:00Z",
        "git_sha": "test",
        "metrics": {"macro_f1": 0.99},
    }
    model.save_pretrained(
        ckpt,
        tokenizer=tiny_tokenizer,
        labels=_FAKE_LABELS,
        manifest=manifest,
    )
    return ckpt


@pytest.fixture(scope="module")
def tiny_routing_yaml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rules_dir = tmp_path_factory.mktemp("routing")
    rules = [
        {
            "intent": label,
            "department": _DEPARTMENTS[i % len(_DEPARTMENTS)],
            "priority": ["P1", "P2", "P3"][i % 3],
            "sla_hours": 1 + i,
            "tags": ["tag_a", "tag_b"],
        }
        for i, label in enumerate(_FAKE_LABELS)
    ]
    routing_path = rules_dir / "routing.yaml"
    routing_path.write_text(yaml.dump(rules), encoding="utf-8")
    return routing_path


@pytest.fixture(scope="module")
def app(tiny_checkpoint: Path, tiny_routing_yaml: Path) -> TestClient:  # type: ignore[misc]
    import os

    os.environ["APP_MODEL_CHECKPOINT_PATH"] = str(tiny_checkpoint)
    os.environ["APP_ROUTING_CONFIG_PATH"] = str(tiny_routing_yaml)
    os.environ["APP_DB_URL"] = "sqlite:///:memory:"
    os.environ["APP_DEVICE"] = "cpu"

    from src.config import get_settings

    get_settings.cache_clear()

    from api.server import app as fastapi_app

    with TestClient(fastapi_app, raise_server_exceptions=True) as client:
        yield client

    get_settings.cache_clear()
    for key in ("APP_MODEL_CHECKPOINT_PATH", "APP_ROUTING_CONFIG_PATH", "APP_DB_URL", "APP_DEVICE"):
        os.environ.pop(key, None)


# ---------- TestAppImport ----------


class TestAppImport:
    def test_app_importable(self) -> None:
        from api.server import app  # noqa: F401

        assert app is not None

    def test_app_has_lifespan(self) -> None:
        from api.server import app, lifespan

        # The app is constructed with our lifespan context manager.
        assert app.router.lifespan_context is lifespan


# ---------- TestLifespanStartup ----------


@pytest.mark.slow
class TestLifespanStartup:
    def test_client_starts_without_error(self, app: TestClient) -> None:
        assert app is not None

    def test_app_state_has_classifier(self, app: TestClient) -> None:
        from src.predict import Classifier

        assert isinstance(app.app.state.classifier, Classifier)  # type: ignore[attr-defined]

    def test_app_state_has_routing_rules(self, app: TestClient) -> None:
        rules = app.app.state.routing_rules  # type: ignore[attr-defined]
        assert isinstance(rules, dict)
        assert len(rules) == _NUM_LABELS

    def test_app_state_has_session_factory(self, app: TestClient) -> None:
        from sqlalchemy.orm import sessionmaker

        assert isinstance(app.app.state.session_factory, sessionmaker)  # type: ignore[attr-defined]

    def test_app_state_has_started_at(self, app: TestClient) -> None:
        from datetime import datetime

        assert isinstance(app.app.state.started_at, datetime)  # type: ignore[attr-defined]


# ---------- TestLifespanFailures ----------


@pytest.mark.slow
class TestLifespanFailures:
    def test_missing_checkpoint_raises_on_startup(
        self, tmp_path: Path, tiny_routing_yaml: Path
    ) -> None:
        import importlib
        import os

        os.environ["APP_MODEL_CHECKPOINT_PATH"] = str(tmp_path / "does_not_exist")
        os.environ["APP_ROUTING_CONFIG_PATH"] = str(tiny_routing_yaml)
        os.environ["APP_DB_URL"] = "sqlite:///:memory:"
        os.environ["APP_DEVICE"] = "cpu"

        from src.config import get_settings

        get_settings.cache_clear()

        import api.server as server_module

        importlib.reload(server_module)

        with (
            pytest.raises(ModelNotLoadedError),
            TestClient(server_module.app, raise_server_exceptions=True),
        ):
            pass

        get_settings.cache_clear()
        for key in (
            "APP_MODEL_CHECKPOINT_PATH",
            "APP_ROUTING_CONFIG_PATH",
            "APP_DB_URL",
            "APP_DEVICE",
        ):
            os.environ.pop(key, None)

    def test_missing_intent_in_routing_raises_on_startup(
        self,
        tmp_path: Path,
        tiny_checkpoint: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        import importlib
        import os

        incomplete = [
            {
                "intent": f"intent_{i:02d}",
                "department": "Cards",
                "priority": "P3",
                "sla_hours": 24,
                "tags": ["a", "b"],
            }
            for i in range(5)
        ]
        incomplete_path = tmp_path / "incomplete_routing.yaml"
        incomplete_path.write_text(yaml.dump(incomplete), encoding="utf-8")

        os.environ["APP_MODEL_CHECKPOINT_PATH"] = str(tiny_checkpoint)
        os.environ["APP_ROUTING_CONFIG_PATH"] = str(incomplete_path)
        os.environ["APP_DB_URL"] = "sqlite:///:memory:"
        os.environ["APP_DEVICE"] = "cpu"

        from src.config import get_settings

        get_settings.cache_clear()

        import api.server as server_module

        importlib.reload(server_module)

        with (
            pytest.raises(ModelNotLoadedError),
            TestClient(server_module.app, raise_server_exceptions=True),
        ):
            pass

        get_settings.cache_clear()
        for key in (
            "APP_MODEL_CHECKPOINT_PATH",
            "APP_ROUTING_CONFIG_PATH",
            "APP_DB_URL",
            "APP_DEVICE",
        ):
            os.environ.pop(key, None)


# ---------- TestDependencyProviders ----------


@pytest.mark.slow
class TestDependencyProviders:
    def test_get_settings_importable(self) -> None:
        from api.server import get_settings_dep  # noqa: F401

    def test_get_classifier_importable(self) -> None:
        from api.server import get_classifier  # noqa: F401

    def test_get_routing_rules_importable(self) -> None:
        from api.server import get_routing_rules  # noqa: F401

    def test_get_session_importable(self) -> None:
        from api.server import get_session  # noqa: F401

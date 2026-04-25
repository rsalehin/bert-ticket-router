"""Tests for T-023: GET /health endpoint."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from transformers import PreTrainedTokenizerBase

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
    ckpt = tmp_path_factory.mktemp("ckpt_health")
    model = BertClassifier(base_model_name=tiny_bert_name, num_labels=_NUM_LABELS)
    manifest = {
        "model_version": "bert-tiny@health-test",
        "base_model": tiny_bert_name,
        "trained_at": "2026-01-01T00:00:00Z",
        "git_sha": "test",
        "metrics": {"macro_f1": 0.99},
    }
    model.save_pretrained(ckpt, tokenizer=tiny_tokenizer, labels=_FAKE_LABELS, manifest=manifest)
    return ckpt


@pytest.fixture(scope="module")
def tiny_routing_yaml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rules_dir = tmp_path_factory.mktemp("routing_health")
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
    p = rules_dir / "routing.yaml"
    p.write_text(yaml.dump(rules), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def client(tiny_checkpoint: Path, tiny_routing_yaml: Path) -> TestClient:  # type: ignore[misc]
    import os

    from src.config import get_settings

    os.environ["APP_MODEL_CHECKPOINT_PATH"] = str(tiny_checkpoint)
    os.environ["APP_ROUTING_CONFIG_PATH"] = str(tiny_routing_yaml)
    os.environ["APP_DB_URL"] = "sqlite:///:memory:"
    os.environ["APP_DEVICE"] = "cpu"
    get_settings.cache_clear()

    from api.server import app

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    get_settings.cache_clear()
    for key in ("APP_MODEL_CHECKPOINT_PATH", "APP_ROUTING_CONFIG_PATH", "APP_DB_URL", "APP_DEVICE"):
        os.environ.pop(key, None)


# ---------- TestHealthEndpoint ----------


@pytest.mark.slow
class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_has_status_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_response_has_model_version(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.json()["model_version"] == "bert-tiny@health-test"

    def test_response_has_device(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.json()["device"] == "cpu"

    def test_response_has_num_labels(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.json()["num_labels"] == _NUM_LABELS

    def test_response_has_uptime_seconds(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.json()["uptime_seconds"] >= 0.0

    def test_uptime_increases(self, client: TestClient) -> None:
        t1 = client.get("/health").json()["uptime_seconds"]
        time.sleep(0.05)
        t2 = client.get("/health").json()["uptime_seconds"]
        assert t2 > t1

    def test_response_has_all_required_fields(self, client: TestClient) -> None:
        body = client.get("/health").json()
        for field in ("status", "model_version", "device", "num_labels", "uptime_seconds"):
            assert field in body, f"missing field: {field}"

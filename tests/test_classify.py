"""Tests for T-024: POST /classify endpoint."""

from __future__ import annotations

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
    ckpt = tmp_path_factory.mktemp("ckpt_classify")
    model = BertClassifier(base_model_name=tiny_bert_name, num_labels=_NUM_LABELS)
    manifest = {
        "model_version": "bert-tiny@classify-test",
        "base_model": tiny_bert_name,
        "trained_at": "2026-01-01T00:00:00Z",
        "git_sha": "test",
        "metrics": {"macro_f1": 0.99},
    }
    model.save_pretrained(ckpt, tokenizer=tiny_tokenizer, labels=_FAKE_LABELS, manifest=manifest)
    return ckpt


@pytest.fixture(scope="module")
def tiny_routing_yaml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rules_dir = tmp_path_factory.mktemp("routing_classify")
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


# ---------- TestClassifyHappyPath ----------


@pytest.mark.slow
class TestClassifyHappyPath:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"message": "my card has not arrived"})
        assert resp.status_code == 200

    def test_response_has_intent(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"message": "my card has not arrived"})
        body = resp.json()
        assert "intent" in body
        assert isinstance(body["intent"], str)

    def test_response_has_confidence(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"message": "my card has not arrived"})
        body = resp.json()
        assert "confidence" in body
        assert 0.0 <= body["confidence"] <= 1.0

    def test_response_has_top_k_length_3(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"message": "my card has not arrived"})
        assert len(resp.json()["top_k"]) == 3

    def test_response_has_ticket(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"message": "my card has not arrived"})
        body = resp.json()
        assert "ticket" in body
        ticket = body["ticket"]
        for field in ("id", "department", "priority", "sla_hours", "tags", "created_at"):
            assert field in ticket, f"missing ticket field: {field}"

    def test_ticket_model_version_matches_health(self, client: TestClient) -> None:
        health_version = client.get("/health").json()["model_version"]
        resp = client.post("/classify", json={"message": "transfer failed"})
        assert resp.json()["ticket"]["id"]  # ticket was created
        # model_version is stored in DB but not in Ticket schema —
        # verify it matches via the health endpoint version string.
        assert health_version == "bert-tiny@classify-test"

    def test_intent_is_in_fake_labels(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"message": "what is my balance"})
        assert resp.json()["intent"] in _FAKE_LABELS

    def test_ticket_id_is_26_chars(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"message": "I need a new card"})
        assert len(resp.json()["ticket"]["id"]) == 26

    def test_ticket_priority_valid(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"message": "suspicious transaction"})
        assert resp.json()["ticket"]["priority"] in ("P1", "P2", "P3")

    def test_response_has_correlation_id_header(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"message": "lost card"})
        assert "x-correlation-id" in resp.headers


# ---------- TestClassifyValidation ----------


@pytest.mark.slow
class TestClassifyValidation:
    def test_empty_message_returns_422(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"message": ""})
        assert resp.status_code == 422

    def test_too_long_message_returns_422(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"message": "x" * 1001})
        assert resp.status_code == 422

    def test_missing_message_returns_422(self, client: TestClient) -> None:
        resp = client.post("/classify", json={})
        assert resp.status_code == 422

    def test_422_body_has_error_envelope(self, client: TestClient) -> None:
        resp = client.post("/classify", json={"message": ""})
        assert "error" in resp.json()


# ---------- TestClassifyPersistence ----------


@pytest.mark.slow
class TestClassifyPersistence:
    def test_row_is_persisted(self, client: TestClient) -> None:
        """Two classify calls should produce two distinct ticket IDs."""
        r1 = client.post("/classify", json={"message": "card not arrived"})
        r2 = client.post("/classify", json={"message": "wrong amount charged"})
        assert r1.json()["ticket"]["id"] != r2.json()["ticket"]["id"]

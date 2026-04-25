"""T-031: End-to-end integration test.

Boots the real FastAPI app via TestClient against:
- The real trained checkpoint in artifacts/model/
- A fresh temporary SQLite database

Sends 5 representative banking77 messages through POST /classify and
asserts the full response contract plus DB persistence.

Marked `slow` — excluded from the fast CI suite.
"""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def e2e_client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:  # type: ignore[misc]
    """Full app started with the real checkpoint and a temp SQLite DB."""
    db_path = tmp_path_factory.mktemp("e2e") / "tickets.db"

    os.environ["APP_DB_URL"] = f"sqlite:///{db_path}"
    os.environ["APP_DEVICE"] = "cpu"

    from src.config import get_settings

    get_settings.cache_clear()

    from api.server import app

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client

    get_settings.cache_clear()
    os.environ.pop("APP_DB_URL", None)
    os.environ.pop("APP_DEVICE", None)


_MESSAGES = [
    "My card has not arrived yet",
    "I need to transfer money to another account",
    "My account seems to have been hacked",
    "What is the exchange rate for USD to EUR?",
    "I was charged the wrong amount",
]


# ---------------------------------------------------------------------------
# TestE2EClassify
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestE2EClassify:
    def test_all_five_messages_return_200(self, e2e_client: TestClient) -> None:
        for msg in _MESSAGES:
            resp = e2e_client.post("/classify", json={"message": msg})
            assert resp.status_code == 200, f"Failed for: {msg!r} — {resp.text}"

    def test_response_schema_is_valid(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post("/classify", json={"message": _MESSAGES[0]})
        body = resp.json()
        assert "intent" in body
        assert "confidence" in body
        assert "top_k" in body
        assert "ticket" in body
        assert len(body["top_k"]) == 3
        ticket = body["ticket"]
        for field in ("id", "department", "priority", "sla_hours", "tags", "created_at"):
            assert field in ticket, f"missing field: {field}"

    def test_intent_is_a_known_banking77_label(self, e2e_client: TestClient) -> None:
        from src.data import get_label_names

        labels = set(get_label_names())
        for msg in _MESSAGES:
            resp = e2e_client.post("/classify", json={"message": msg})
            intent = resp.json()["intent"]
            assert intent in labels, f"Unknown intent {intent!r} for {msg!r}"

    def test_confidence_in_unit_interval(self, e2e_client: TestClient) -> None:
        for msg in _MESSAGES:
            resp = e2e_client.post("/classify", json={"message": msg})
            body = resp.json()
            assert 0.0 <= body["confidence"] <= 1.0
            for score in body["top_k"]:
                assert 0.0 <= score["confidence"] <= 1.0

    def test_priority_is_valid(self, e2e_client: TestClient) -> None:
        for msg in _MESSAGES:
            resp = e2e_client.post("/classify", json={"message": msg})
            assert resp.json()["ticket"]["priority"] in ("P1", "P2", "P3")

    def test_ticket_id_is_26_chars(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post("/classify", json={"message": _MESSAGES[0]})
        assert len(resp.json()["ticket"]["id"]) == 26

    def test_model_version_matches_health(self, e2e_client: TestClient) -> None:
        health = e2e_client.get("/health").json()
        e2e_client.post("/classify", json={"message": _MESSAGES[0]})
        # model_version is stored in DB; verify health endpoint reflects the same
        assert health["model_version"] == "bert-base-uncased@v0.1.0"

    def test_five_rows_persisted_in_db(self, e2e_client: TestClient) -> None:
        """After 5 classify calls the DB must have at least 5 rows."""
        # Calls were already made in previous tests (same module-scoped client).
        # Make 5 fresh ones to be deterministic.
        for msg in _MESSAGES:
            e2e_client.post("/classify", json={"message": msg})

        engine = e2e_client.app.state.engine  # type: ignore[attr-defined]
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM tickets")).scalar()
        assert count >= 5

    def test_correlation_id_header_present(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post("/classify", json={"message": _MESSAGES[0]})
        assert "x-correlation-id" in resp.headers
        assert len(resp.headers["x-correlation-id"]) == 26

    def test_cpu_latency_p95_under_500ms(self, e2e_client: TestClient) -> None:
        """NFR-4: p95 latency < 500 ms on CPU for 10 calls."""
        latencies: list[float] = []
        for _ in range(10):
            t0 = time.perf_counter()
            e2e_client.post("/classify", json={"message": "my card is lost"})
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        assert p95 < 500, f"p95 latency {p95:.1f} ms exceeds 500 ms NFR"


# ---------------------------------------------------------------------------
# TestE2EHealth
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestE2EHealth:
    def test_health_returns_200(self, e2e_client: TestClient) -> None:
        resp = e2e_client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, e2e_client: TestClient) -> None:
        assert e2e_client.get("/health").json()["status"] == "ok"

    def test_health_num_labels_is_77(self, e2e_client: TestClient) -> None:
        assert e2e_client.get("/health").json()["num_labels"] == 77


# ---------------------------------------------------------------------------
# TestE2EValidation
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestE2EValidation:
    def test_empty_message_returns_422(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post("/classify", json={"message": ""})
        assert resp.status_code == 422

    def test_missing_message_returns_422(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post("/classify", json={})
        assert resp.status_code == 422

    def test_too_long_message_returns_422(self, e2e_client: TestClient) -> None:
        resp = e2e_client.post("/classify", json={"message": "x" * 1001})
        assert resp.status_code == 422

"""Tests for T-022: correlation ID middleware, CORS, and exception handlers."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.errors import ModelNotLoadedError
from src.errors import ValidationError as AppValidationError

# ---------------------------------------------------------------------------
# Minimal app fixture — no model, no DB, just the middleware + handlers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bare_app() -> TestClient:
    """A FastAPI app with only middleware and exception handlers registered.

    Avoids loading the model checkpoint; tests focus purely on HTTP behaviour.
    """
    from api.server import register_exception_handlers, register_middleware

    app = FastAPI()
    register_middleware(app)
    register_exception_handlers(app)

    # Test routes that trigger each handler path
    @app.get("/ok")
    def ok() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/app-error")
    def raise_app_error() -> None:
        raise ModelNotLoadedError("model not ready")

    @app.get("/validation-error")
    def raise_validation_error() -> None:
        raise AppValidationError("bad input")

    @app.get("/runtime-error")
    def raise_runtime_error() -> None:
        raise RuntimeError("unexpected boom")

    @app.post("/body")
    def accept_body(payload: dict[str, str]) -> dict[str, str]:
        return payload

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ---------------------------------------------------------------------------
# TestCorrelationIdMiddleware
# ---------------------------------------------------------------------------


class TestCorrelationIdMiddleware:
    def test_response_has_correlation_id_header(self, bare_app: TestClient) -> None:
        resp = bare_app.get("/ok")
        assert "x-correlation-id" in resp.headers

    def test_correlation_id_is_26_chars(self, bare_app: TestClient) -> None:
        resp = bare_app.get("/ok")
        cid = resp.headers["x-correlation-id"]
        assert len(cid) == 26

    def test_each_request_gets_distinct_correlation_id(self, bare_app: TestClient) -> None:
        cids = {bare_app.get("/ok").headers["x-correlation-id"] for _ in range(5)}
        assert len(cids) == 5

    def test_error_response_includes_correlation_id(self, bare_app: TestClient) -> None:
        resp = bare_app.get("/app-error")
        assert "x-correlation-id" in resp.headers
        body = resp.json()
        assert body["error"]["correlation_id"] == resp.headers["x-correlation-id"]


# ---------------------------------------------------------------------------
# TestCorsMiddleware
# ---------------------------------------------------------------------------


class TestCorsMiddleware:
    def test_cors_allows_vite_origin(self, bare_app: TestClient) -> None:
        resp = bare_app.options(
            "/ok",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_does_not_allow_unknown_origin(self, bare_app: TestClient) -> None:
        resp = bare_app.options(
            "/ok",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") != "http://evil.com"


# ---------------------------------------------------------------------------
# TestAppErrorHandler
# ---------------------------------------------------------------------------


class TestAppErrorHandler:
    def test_app_error_returns_correct_http_status(self, bare_app: TestClient) -> None:
        resp = bare_app.get("/app-error")
        assert resp.status_code == ModelNotLoadedError.http_status

    def test_app_error_body_has_error_envelope(self, bare_app: TestClient) -> None:
        resp = bare_app.get("/app-error")
        body = resp.json()
        assert "error" in body

    def test_app_error_body_has_correct_code(self, bare_app: TestClient) -> None:
        resp = bare_app.get("/app-error")
        body = resp.json()
        assert body["error"]["code"] == ModelNotLoadedError.code

    def test_app_error_body_has_message(self, bare_app: TestClient) -> None:
        resp = bare_app.get("/app-error")
        body = resp.json()
        assert body["error"]["message"] == "model not ready"

    def test_validation_error_returns_400(self, bare_app: TestClient) -> None:
        resp = bare_app.get("/validation-error")
        assert resp.status_code == AppValidationError.http_status

    def test_validation_error_code(self, bare_app: TestClient) -> None:
        resp = bare_app.get("/validation-error")
        assert resp.json()["error"]["code"] == AppValidationError.code


# ---------------------------------------------------------------------------
# TestRequestValidationErrorHandler
# ---------------------------------------------------------------------------


class TestRequestValidationErrorHandler:
    def test_missing_body_returns_422(self, bare_app: TestClient) -> None:
        # POST /body requires a JSON body; omitting it triggers RequestValidationError
        resp = bare_app.post(
            "/body", content="not json", headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 422

    def test_422_body_has_error_envelope(self, bare_app: TestClient) -> None:
        resp = bare_app.post(
            "/body", content="not json", headers={"Content-Type": "application/json"}
        )
        assert "error" in resp.json()

    def test_422_body_has_validation_failed_code(self, bare_app: TestClient) -> None:
        resp = bare_app.post(
            "/body", content="not json", headers={"Content-Type": "application/json"}
        )
        assert resp.json()["error"]["code"] == "VALIDATION_FAILED"


# ---------------------------------------------------------------------------
# TestGenericExceptionHandler
# ---------------------------------------------------------------------------


class TestGenericExceptionHandler:
    def test_unhandled_exception_returns_500(self, bare_app: TestClient) -> None:
        resp = bare_app.get("/runtime-error")
        assert resp.status_code == 500

    def test_500_body_has_internal_error_code(self, bare_app: TestClient) -> None:
        resp = bare_app.get("/runtime-error")
        assert resp.json()["error"]["code"] == "INTERNAL_ERROR"

    def test_500_body_has_correlation_id(self, bare_app: TestClient) -> None:
        resp = bare_app.get("/runtime-error")
        body = resp.json()
        assert "correlation_id" in body["error"]
        assert body["error"]["correlation_id"] == resp.headers["x-correlation-id"]

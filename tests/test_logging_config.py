"""Tests for src.logging_config (structlog configuration helpers)."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator

import pytest
import structlog

from src.logging_config import bind_correlation_id, clear_correlation_id, configure_logging


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Reset structlog config and contextvars between tests."""
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    yield
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


def _capture_log_output(level: str = "INFO") -> io.StringIO:
    """Configure logging to write JSON to a StringIO and return it."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    configure_logging(level)
    return buf


class TestConfigureLogging:
    def test_runs_without_error(self) -> None:
        configure_logging("INFO")

    def test_accepts_all_documented_levels(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            configure_logging(level)

    def test_emits_json_with_required_keys(self) -> None:
        buf = _capture_log_output("INFO")
        log = structlog.get_logger()
        log.info("hello_world", foo="bar")

        line = buf.getvalue().strip().splitlines()[-1]
        record = json.loads(line)

        assert record["event"] == "hello_world"
        assert record["level"] == "info"
        assert "timestamp" in record
        assert record["foo"] == "bar"

    def test_respects_log_level(self) -> None:
        buf = _capture_log_output("WARNING")
        log = structlog.get_logger()
        log.info("should_not_appear")
        log.warning("should_appear")

        output = buf.getvalue()
        assert "should_not_appear" not in output
        assert "should_appear" in output


class TestCorrelationId:
    def test_bind_adds_correlation_id_to_subsequent_logs(self) -> None:
        buf = _capture_log_output("INFO")
        bind_correlation_id("01HXR3F8Z4G2N6P7Q9S0T2U4V6")

        log = structlog.get_logger()
        log.info("with_cid")

        line = buf.getvalue().strip().splitlines()[-1]
        record = json.loads(line)
        assert record["correlation_id"] == "01HXR3F8Z4G2N6P7Q9S0T2U4V6"

    def test_clear_removes_correlation_id(self) -> None:
        buf = _capture_log_output("INFO")
        bind_correlation_id("01HXR3F8Z4G2N6P7Q9S0T2U4V6")
        clear_correlation_id()

        log = structlog.get_logger()
        log.info("after_clear")

        line = buf.getvalue().strip().splitlines()[-1]
        record = json.loads(line)
        assert "correlation_id" not in record

    def test_no_correlation_id_when_unbound(self) -> None:
        buf = _capture_log_output("INFO")
        log = structlog.get_logger()
        log.info("without_cid")

        line = buf.getvalue().strip().splitlines()[-1]
        record = json.loads(line)
        assert "correlation_id" not in record

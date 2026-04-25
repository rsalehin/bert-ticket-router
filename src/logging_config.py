"""Structured logging configuration for the application.

Wraps `structlog` to emit JSON log records with `level`, `event`, `timestamp`
keys plus arbitrary structured context. Used by the FastAPI server (called in
the lifespan startup) and by `train.py` (called at the top of `train()`).

Correlation IDs are stored in `structlog.contextvars`, so they propagate
within the same async task / thread context. Bind one at the start of a
request; clear it at the end. Outside a request (e.g. training), correlation
IDs are unused.
"""

from __future__ import annotations

import logging
import sys

import structlog

LogLevel = str  # narrowed by Settings.log_level (Literal["DEBUG", "INFO", "WARNING", "ERROR"])


def configure_logging(level: LogLevel) -> None:
    """Configure stdlib logging and structlog to emit JSON to stdout.

    Idempotent: safe to call more than once. The given `level` is applied
    to the root logger and to structlog's filtering.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level: {level!r}")

    # Stdlib logging: write to stdout with a plain formatter; structlog's
    # JSON renderer produces the actual structured payload, so the stdlib
    # formatter only needs to pass the message through.
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
    root.setLevel(numeric_level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_correlation_id(correlation_id: str) -> None:
    """Bind a correlation ID to the current logging context.

    Subsequent log records emitted in the same context include it as
    `correlation_id`. Typically called at the start of a request.
    """
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def clear_correlation_id() -> None:
    """Remove the correlation ID from the current logging context.

    Typically called at the end of a request.
    """
    structlog.contextvars.unbind_contextvars("correlation_id")

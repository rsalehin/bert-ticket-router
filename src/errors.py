"""Domain exception hierarchy for the ticket router.

All application-level errors derive from `AppError`. Each subclass carries
class-level `code` (machine-readable, for the API error response body) and
`http_status` (default HTTP status code used by the FastAPI exception handler).

The API layer pattern-matches on `AppError` to translate domain errors into
structured HTTP responses. Anything outside this hierarchy is treated as an
unexpected internal error (500) by the fallback handler.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all domain errors raised by this application."""

    code: str = "APP_ERROR"
    http_status: int = 500


class ValidationError(AppError):
    """Application-level validation failure (beyond Pydantic's request validation)."""

    code: str = "VALIDATION_ERROR"
    http_status: int = 400


class IntentNotInRoutingError(AppError):
    """Predicted intent has no matching rule in the routing config."""

    code: str = "INTENT_NOT_IN_ROUTING"
    http_status: int = 500


class ModelNotLoadedError(AppError):
    """Inference attempted before the model is loaded, or checkpoint is incomplete."""

    code: str = "MODEL_NOT_LOADED"
    http_status: int = 503


class PersistenceError(AppError):
    """Database operation failed (connection, write, or read)."""

    code: str = "PERSISTENCE_ERROR"
    http_status: int = 503

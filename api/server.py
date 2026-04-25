"""FastAPI application: construction, lifespan, middleware, and exception handlers."""

from __future__ import annotations

import traceback
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import structlog
import structlog.contextvars as ctx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
from ulid import ULID

from src.config import get_settings
from src.errors import AppError, IntentNotInRoutingError
from src.logging_config import configure_logging
from src.persistence import init_db, make_engine, make_session
from src.predict import Classifier
from src.routing import load_routing_rules
from src.schemas import ClassifyRequest, ClassifyResponse, RoutingRule

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Generate a ULID correlation ID per request, bind to structlog context."""

    async def dispatch(
        self, request: Request, call_next: Callable[..., Awaitable[StarletteResponse]]
    ) -> StarletteResponse:
        cid = str(ULID())
        ctx.clear_contextvars()
        ctx.bind_contextvars(correlation_id=cid)

        try:
            response: StarletteResponse = await call_next(request)
        except Exception:
            # Unhandled exceptions that escape call_next (e.g. RuntimeError in
            # sync route handlers run in threadpool) won't reach the FastAPI
            # exception handler *and* set a response header. Build the 500
            # response here so the correlation ID is always present.
            response = JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "An unexpected error occurred",
                        "correlation_id": cid,
                    }
                },
            )

        response.headers["X-Correlation-ID"] = cid
        return response


def register_middleware(app: FastAPI) -> None:
    """Attach CORS and correlation-ID middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # CorrelationIdMiddleware wraps CORS ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â added last, runs first.
    app.add_middleware(CorrelationIdMiddleware)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


def register_exception_handlers(app: FastAPI) -> None:
    """Register AppError, RequestValidationError, and generic 500 handlers."""
    log = structlog.get_logger(__name__)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        bound = ctx.get_contextvars()
        cid = str(bound.get("correlation_id", ""))
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": {
                    "code": exc.code,
                    "message": str(exc),
                    "correlation_id": cid,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        bound = ctx.get_contextvars()
        cid = str(bound.get("correlation_id", ""))
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_FAILED",
                    "message": "Request validation failed",
                    "correlation_id": cid,
                    "details": exc.errors(),
                }
            },
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
        bound = ctx.get_contextvars()
        cid = str(bound.get("correlation_id", ""))
        log.error(
            "unhandled_exception",
            exc_type=type(exc).__name__,
            exc=str(exc),
            traceback=traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "correlation_id": cid,
                }
            },
        )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load all resources at startup; release on shutdown."""
    settings = get_settings()

    configure_logging(settings.log_level)
    log = structlog.get_logger(__name__)
    log.info("server_startup", device=settings.device)

    routing_rules: dict[str, RoutingRule] = load_routing_rules(settings.routing_config_path)
    log.info("routing_rules_loaded", count=len(routing_rules))

    classifier = Classifier(
        checkpoint_path=settings.model_checkpoint_path,
        device=settings.device,
        max_len=settings.max_len,
    )
    log.info(
        "classifier_loaded",
        model_version=classifier.model_version,
        num_labels=classifier.num_labels,
        device=classifier.device,
    )

    missing = [
        label
        for label in classifier._labels  # noqa: SLF001
        if label not in routing_rules
    ]
    if missing:
        raise IntentNotInRoutingError(
            f"Classifier labels missing from routing rules: {missing[:5]!r}"
            + (" (and more)" if len(missing) > 5 else "")
        )

    engine = make_engine(settings.db_url)
    session_factory = make_session(engine)
    init_db(engine)
    log.info("database_ready", db_url=settings.db_url)

    app.state.classifier = classifier
    app.state.routing_rules = routing_rules
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.started_at = datetime.now(UTC)

    yield

    log.info("server_shutdown")


# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------


app = FastAPI(
    title="BERT Ticket Router",
    version="0.1.0",
    description="Classify customer messages and route them to the right department.",
    lifespan=lifespan,
)

register_middleware(app)
register_exception_handlers(app)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    """Return server and model status."""
    from datetime import UTC, datetime

    classifier: Classifier = request.app.state.classifier
    started_at: datetime = request.app.state.started_at
    uptime = (datetime.now(UTC) - started_at).total_seconds()

    return {
        "status": "ok",
        "model_version": classifier.model_version,
        "device": classifier.device,
        "num_labels": classifier.num_labels,
        "uptime_seconds": round(uptime, 3),
    }


@app.post("/classify")
def classify_message(
    request: Request,
    body: ClassifyRequest,
) -> ClassifyResponse:
    """Classify a customer message and persist the resulting ticket."""
    import time as _time

    import structlog as _structlog

    from src.persistence import save_ticket
    from src.routing import build_ticket
    from src.schemas import ClassifyResponse

    _log = _structlog.get_logger(__name__)
    t0 = _time.perf_counter()

    classifier: Classifier = request.app.state.classifier
    rules: dict[str, RoutingRule] = request.app.state.routing_rules
    session_factory = request.app.state.session_factory

    prediction = classifier.classify(body.message)
    ticket = build_ticket(prediction, rules, classifier.model_version)

    with session_factory() as session:
        saved = save_ticket(
            session,
            ticket,
            message=body.message,
            prediction=prediction,
            model_version=classifier.model_version,
        )

    latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    _log.info(
        "classify",
        intent=prediction.intent,
        confidence=prediction.confidence,
        latency_ms=latency_ms,
    )

    return ClassifyResponse(
        intent=prediction.intent,
        confidence=prediction.confidence,
        top_k=prediction.top_k,
        ticket=saved,
    )

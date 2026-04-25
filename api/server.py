"""FastAPI application: construction, lifespan, and dependency providers."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import structlog
from fastapi import FastAPI, Request
from sqlalchemy.orm import Session

from src.config import Settings, get_settings
from src.errors import IntentNotInRoutingError
from src.logging_config import configure_logging
from src.persistence import init_db, make_engine, make_session
from src.predict import Classifier
from src.routing import load_routing_rules
from src.schemas import RoutingRule

logger = structlog.get_logger(__name__)


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


app = FastAPI(
    title="BERT Ticket Router",
    version="0.1.0",
    description="Classify customer messages and route them to the right department.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------


def get_settings_dep() -> Settings:
    return get_settings()


def get_classifier(request: Request) -> Classifier:
    return request.app.state.classifier  # type: ignore[no-any-return]


def get_routing_rules(request: Request) -> dict[str, RoutingRule]:
    return request.app.state.routing_rules  # type: ignore[no-any-return]


def get_session(request: Request) -> Generator[Session, None, None]:
    with request.app.state.session_factory() as session:
        yield session

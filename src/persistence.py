"""Persistence layer: SQLAlchemy ORM, engine + session factories, DB init.

This module owns the database schema (`TicketRow`) and the machinery to
create engines and sessions. CRUD operations (`save_ticket`, `list_tickets`,
`get_ticket`) live in T-019.

The architecture distinguishes:
- `Ticket` (Pydantic, in `src/schemas.py`)         -> API and domain shape.
- `TicketRow` (SQLAlchemy, here)                   -> database row mapping.

`top_k` and `tags` are stored as JSON-in-TEXT (`top_k_json`, `tags_json`)
because SQLite has no native array type. The conversion between Pydantic
and ORM happens at the persistence-layer boundary in T-019; the rest of
the codebase only sees `Ticket`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal, cast

from sqlalchemy import (
    DateTime,
    Engine,
    Float,
    Index,
    Integer,
    String,
    TypeDecorator,
    create_engine,
    select,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.errors import PersistenceError
from src.schemas import Prediction, Ticket


class UtcDateTime(TypeDecorator[datetime]):
    """A DateTime column that always returns timezone-aware UTC datetimes.

    SQLite stores datetimes as ISO text without offset; SQLAlchemy's
    `DateTime(timezone=True)` therefore returns a naive `datetime` on read.
    This decorator restores tz-awareness so callers can rely on it.

    On write: incoming naive datetimes are rejected (we want loud failures
    for any code path that forgets tz). UTC offsets pass through.
    On read: naive results from the DB are stamped with `UTC`.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime not allowed; pass a timezone-aware UTC datetime")
        return value.astimezone(UTC)

    def process_result_value(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    """Declarative base for all ORM models in this project."""


class TicketRow(Base):
    """SQLAlchemy 2.x typed model for the `tickets` table.

    Mirrors the schema in `docs/architecture.md` section 5. Indexes:
        - ix_tickets_created_at : recent-tickets queries
        - ix_tickets_department : filter by department
        - ix_tickets_intent     : filter by intent
    """

    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    message: Mapped[str] = mapped_column(String, nullable=False)
    intent: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    top_k_json: Mapped[str] = mapped_column(String, nullable=False)
    department: Mapped[str] = mapped_column(String, nullable=False)
    priority: Mapped[str] = mapped_column(String, nullable=False)
    sla_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    tags_json: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_tickets_created_at", "created_at"),
        Index("ix_tickets_department", "department"),
        Index("ix_tickets_intent", "intent"),
    )


def make_engine(url: str) -> Engine:
    """Create a SQLAlchemy engine for the given DB URL.

    For SQLite, enables `check_same_thread=False` so a single engine can be
    safely shared across FastAPI's threadpool. Concurrency safety is provided
    by per-request sessions (T-019), not by SQLite's default locking.
    """
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, future=True, connect_args=connect_args)


def make_session(engine: Engine) -> sessionmaker[Session]:
    """Build a sessionmaker bound to the given engine.

    The returned factory is suitable for use as a FastAPI dependency:
    one session per request, closed at request end.
    """
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Idempotently create all tables on the given engine.

    Equivalent to running the Alembic baseline migration but faster ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â used
    in tests and as a first-run convenience. Production deploys should use
    `alembic upgrade head` for proper migration tracking.
    """
    Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


def _row_to_ticket(row: TicketRow) -> Ticket:
    """Convert a SQLAlchemy `TicketRow` to a Pydantic `Ticket`.

    Only the routing fields are surfaced; audit columns (message, intent,
    confidence, top_k_json, model_version) remain in the DB and are not
    exposed via the persistence API.
    """
    tags: list[str] = json.loads(row.tags_json)
    return Ticket(
        id=row.id,
        department=row.department,
        priority=cast(Literal["P1", "P2", "P3"], row.priority),
        sla_hours=row.sla_hours,
        tags=tags,
        created_at=row.created_at,
    )


def save_ticket(
    session: Session,
    ticket: Ticket,
    *,
    message: str,
    prediction: Prediction,
    model_version: str,
) -> Ticket:
    """Persist a ticket plus the prediction context that produced it.

    Args:
        session: An active SQLAlchemy session.
        ticket: The Pydantic `Ticket` (routing fields, id, created_at).
        message: The original customer message; stored on the audit column.
        prediction: The full classifier prediction; `top_k` and `intent`
            and `confidence` are stored as audit columns.
        model_version: The `model_version` string from the classifier; stored
            on the audit column.

    Returns:
        The same `Ticket` back, after a successful commit.

    Raises:
        PersistenceError: If the database operation fails.
    """
    top_k_payload = [score.model_dump() for score in prediction.top_k]
    row = TicketRow(
        id=ticket.id,
        message=message,
        intent=prediction.intent,
        confidence=prediction.confidence,
        top_k_json=json.dumps(top_k_payload),
        department=ticket.department,
        priority=ticket.priority,
        sla_hours=ticket.sla_hours,
        tags_json=json.dumps(list(ticket.tags)),
        model_version=model_version,
        created_at=ticket.created_at,
    )
    try:
        session.add(row)
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        raise PersistenceError(f"Failed to save ticket {ticket.id!r}: {exc}") from exc

    return ticket


def get_ticket(session: Session, ticket_id: str) -> Ticket | None:
    """Look up a ticket by id. Returns `None` if not found."""
    row = session.get(TicketRow, ticket_id)
    if row is None:
        return None
    return _row_to_ticket(row)


def list_tickets(session: Session, limit: int = 50) -> list[Ticket]:
    """Return the `limit` most recently created tickets, newest first."""
    stmt = select(TicketRow).order_by(TicketRow.created_at.desc()).limit(limit)
    rows = session.execute(stmt).scalars().all()
    return [_row_to_ticket(row) for row in rows]

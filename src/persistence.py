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

from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    Engine,
    Float,
    Index,
    Integer,
    String,
    TypeDecorator,
    create_engine,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


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

    Equivalent to running the Alembic baseline migration but faster Ã¢â‚¬â€ used
    in tests and as a first-run convenience. Production deploys should use
    `alembic upgrade head` for proper migration tracking.
    """
    Base.metadata.create_all(engine)

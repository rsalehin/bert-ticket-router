"""Tests for src.persistence — ORM model, factories, init_db, and CRUD.

Covers T-018 (ORM + factories + init_db + Alembic baseline) and T-019 (CRUD).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from src.errors import PersistenceError
from src.persistence import (
    Base,
    TicketRow,
    get_ticket,
    init_db,
    list_tickets,
    make_engine,
    make_session,
    save_ticket,
)
from src.schemas import IntentScore, Prediction, Ticket

# ---------- make_engine ----------


class TestMakeEngine:
    def test_returns_engine(self) -> None:
        engine = make_engine("sqlite:///:memory:")
        assert isinstance(engine, Engine)

    def test_engine_url_is_set(self) -> None:
        engine = make_engine("sqlite:///:memory:")
        assert "sqlite" in str(engine.url)

    def test_file_url_works(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = make_engine(f"sqlite:///{db_path}")
        assert isinstance(engine, Engine)


# ---------- make_session ----------


class TestMakeSession:
    def test_returns_sessionmaker(self) -> None:
        engine = make_engine("sqlite:///:memory:")
        factory = make_session(engine)
        assert isinstance(factory, sessionmaker)

    def test_sessionmaker_produces_sessions(self) -> None:
        engine = make_engine("sqlite:///:memory:")
        factory = make_session(engine)
        with factory() as session:
            assert isinstance(session, Session)


# ---------- init_db ----------


class TestInitDb:
    def test_creates_tickets_table(self) -> None:
        engine = make_engine("sqlite:///:memory:")
        init_db(engine)
        inspector = inspect(engine)
        assert "tickets" in inspector.get_table_names()

    def test_is_idempotent(self) -> None:
        engine = make_engine("sqlite:///:memory:")
        init_db(engine)
        init_db(engine)
        inspector = inspect(engine)
        assert "tickets" in inspector.get_table_names()


# ---------- TicketRow schema ----------


class TestTicketRowColumns:
    @pytest.fixture
    def inspector(self) -> object:
        engine = make_engine("sqlite:///:memory:")
        init_db(engine)
        return inspect(engine)

    def test_all_required_columns_present(self, inspector: object) -> None:
        cols = {c["name"] for c in inspector.get_columns("tickets")}  # type: ignore[attr-defined]
        expected = {
            "id",
            "message",
            "intent",
            "confidence",
            "top_k_json",
            "department",
            "priority",
            "sla_hours",
            "tags_json",
            "model_version",
            "created_at",
        }
        assert cols == expected

    def test_id_is_primary_key(self, inspector: object) -> None:
        pk = inspector.get_pk_constraint("tickets")  # type: ignore[attr-defined]
        assert pk["constrained_columns"] == ["id"]

    def test_all_columns_nullable_false(self, inspector: object) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("tickets")}  # type: ignore[attr-defined]
        for name in (
            "id",
            "message",
            "intent",
            "confidence",
            "top_k_json",
            "department",
            "priority",
            "sla_hours",
            "tags_json",
            "model_version",
            "created_at",
        ):
            assert cols[name]["nullable"] is False, f"{name} should be NOT NULL"


class TestTicketRowIndexes:
    @pytest.fixture
    def inspector(self) -> object:
        engine = make_engine("sqlite:///:memory:")
        init_db(engine)
        return inspect(engine)

    def test_index_on_created_at(self, inspector: object) -> None:
        names = {ix["name"] for ix in inspector.get_indexes("tickets")}  # type: ignore[attr-defined]
        assert "ix_tickets_created_at" in names

    def test_index_on_department(self, inspector: object) -> None:
        names = {ix["name"] for ix in inspector.get_indexes("tickets")}  # type: ignore[attr-defined]
        assert "ix_tickets_department" in names

    def test_index_on_intent(self, inspector: object) -> None:
        names = {ix["name"] for ix in inspector.get_indexes("tickets")}  # type: ignore[attr-defined]
        assert "ix_tickets_intent" in names


# ---------- TicketRow round-trip ----------


class TestTicketRowRoundTrip:
    def test_insert_and_query(self) -> None:
        engine = make_engine("sqlite:///:memory:")
        init_db(engine)
        factory = make_session(engine)

        row = TicketRow(
            id="01HXR3F8Z4G2N6P7Q9S0T2U4V6",
            message="My card has not arrived",
            intent="card_arrival",
            confidence=0.94,
            top_k_json='[{"intent":"card_arrival","confidence":0.94}]',
            department="Cards",
            priority="P3",
            sla_hours=24,
            tags_json='["card","delivery"]',
            model_version="bert@v0.1.0",
            created_at=datetime.now(UTC),
        )

        with factory() as session:
            session.add(row)
            session.commit()

        with factory() as session:
            loaded = session.get(TicketRow, "01HXR3F8Z4G2N6P7Q9S0T2U4V6")
            assert loaded is not None
            assert loaded.intent == "card_arrival"
            assert loaded.priority == "P3"
            assert loaded.sla_hours == 24
            assert loaded.created_at.tzinfo is not None


# ---------- Base ----------


class TestBase:
    def test_base_has_metadata(self) -> None:
        assert Base.metadata is not None
        assert "tickets" in Base.metadata.tables


# ---------- Alembic baseline migration ----------


@pytest.mark.slow
class TestAlembicBaselineMigration:
    """`alembic upgrade head` against a fresh DB must produce the ORM schema."""

    def _run_upgrade(self, db_path: Path) -> None:
        import os
        import subprocess
        import sys

        env = os.environ.copy()
        env["APP_DB_URL"] = f"sqlite:///{db_path}"
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).resolve().parent.parent),
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"alembic upgrade head failed: {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

    def test_upgrade_creates_tickets_table(self, tmp_path: Path) -> None:
        db_path = tmp_path / "alembic_test.db"
        self._run_upgrade(db_path)

        engine = make_engine(f"sqlite:///{db_path}")
        cols = {c["name"] for c in inspect(engine).get_columns("tickets")}
        expected = {
            "id",
            "message",
            "intent",
            "confidence",
            "top_k_json",
            "department",
            "priority",
            "sla_hours",
            "tags_json",
            "model_version",
            "created_at",
        }
        assert cols == expected

    def test_upgrade_creates_indexes(self, tmp_path: Path) -> None:
        db_path = tmp_path / "alembic_test.db"
        self._run_upgrade(db_path)

        engine = make_engine(f"sqlite:///{db_path}")
        names = {ix["name"] for ix in inspect(engine).get_indexes("tickets")}
        for required in (
            "ix_tickets_created_at",
            "ix_tickets_department",
            "ix_tickets_intent",
        ):
            assert required in names

    def test_alembic_schema_matches_orm_schema(self, tmp_path: Path) -> None:
        db_alembic = tmp_path / "alembic.db"
        self._run_upgrade(db_alembic)
        eng_a = make_engine(f"sqlite:///{db_alembic}")
        cols_a = {c["name"]: c["nullable"] for c in inspect(eng_a).get_columns("tickets")}
        idx_a = {ix["name"] for ix in inspect(eng_a).get_indexes("tickets")}

        db_orm = tmp_path / "orm.db"
        eng_b = make_engine(f"sqlite:///{db_orm}")
        init_db(eng_b)
        cols_b = {c["name"]: c["nullable"] for c in inspect(eng_b).get_columns("tickets")}
        idx_b = {ix["name"] for ix in inspect(eng_b).get_indexes("tickets")}

        assert cols_a == cols_b, f"column mismatch: alembic={cols_a}, orm={cols_b}"
        assert idx_a == idx_b, f"index mismatch: alembic={idx_a}, orm={idx_b}"


# ---------- CRUD helpers and fixtures ----------


def _sample_ticket(
    *,
    ticket_id: str = "01HXR3F8Z4G2N6P7Q9S0T2U4V6",
    created_at: datetime | None = None,
    department: str = "Cards",
    priority: str = "P3",
    sla_hours: int = 24,
    tags: list[str] | None = None,
) -> Ticket:
    return Ticket(
        id=ticket_id,
        department=department,
        priority=priority,  # type: ignore[arg-type]
        sla_hours=sla_hours,
        tags=tags if tags is not None else ["card", "delivery"],
        created_at=created_at if created_at is not None else datetime.now(UTC),
    )


def _sample_prediction() -> Prediction:
    return Prediction(
        intent="card_arrival",
        confidence=0.94,
        top_k=[
            IntentScore(intent="card_arrival", confidence=0.94),
            IntentScore(intent="card_delivery_estimate", confidence=0.04),
            IntentScore(intent="lost_or_stolen_card", confidence=0.02),
        ],
    )


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    return make_session(engine)


# ---------- save_ticket ----------


class TestSaveTicketHappyPath:
    def test_returns_ticket_pydantic(self, session_factory: sessionmaker[Session]) -> None:
        ticket = _sample_ticket()
        with session_factory() as session:
            saved = save_ticket(
                session,
                ticket,
                message="My card has not arrived",
                prediction=_sample_prediction(),
                model_version="bert@v0.1.0",
            )
        assert isinstance(saved, Ticket)
        assert saved.id == ticket.id
        assert saved.department == "Cards"
        assert saved.priority == "P3"
        assert saved.sla_hours == 24
        assert saved.tags == ["card", "delivery"]

    def test_persists_row_in_db(self, session_factory: sessionmaker[Session]) -> None:
        ticket = _sample_ticket()
        with session_factory() as session:
            save_ticket(
                session,
                ticket,
                message="My card has not arrived",
                prediction=_sample_prediction(),
                model_version="bert@v0.1.0",
            )

        with session_factory() as session:
            row = session.get(TicketRow, ticket.id)
            assert row is not None
            assert row.message == "My card has not arrived"
            assert row.intent == "card_arrival"
            assert row.confidence == pytest.approx(0.94)
            assert row.model_version == "bert@v0.1.0"

    def test_serializes_top_k_as_json(self, session_factory: sessionmaker[Session]) -> None:
        import json as _json

        ticket = _sample_ticket()
        with session_factory() as session:
            save_ticket(
                session,
                ticket,
                message="...",
                prediction=_sample_prediction(),
                model_version="bert@v0.1.0",
            )

        with session_factory() as session:
            row = session.get(TicketRow, ticket.id)
            assert row is not None
            top_k = _json.loads(row.top_k_json)
            assert isinstance(top_k, list)
            assert len(top_k) == 3
            assert top_k[0]["intent"] == "card_arrival"
            assert top_k[0]["confidence"] == pytest.approx(0.94)

    def test_serializes_tags_as_json(self, session_factory: sessionmaker[Session]) -> None:
        import json as _json

        ticket = _sample_ticket(tags=["foo", "bar", "baz"])
        with session_factory() as session:
            save_ticket(
                session,
                ticket,
                message="...",
                prediction=_sample_prediction(),
                model_version="bert@v0.1.0",
            )

        with session_factory() as session:
            row = session.get(TicketRow, ticket.id)
            assert row is not None
            tags = _json.loads(row.tags_json)
            assert tags == ["foo", "bar", "baz"]


class TestSaveTicketCommitFailure:
    def test_raises_persistence_error_when_table_missing(self) -> None:
        """Engine without the table -> commit fails -> PersistenceError."""
        engine = make_engine("sqlite:///:memory:")
        # Note: NOT calling init_db(); the `tickets` table does not exist.
        factory = make_session(engine)

        with pytest.raises(PersistenceError), factory() as session:
            save_ticket(
                session,
                _sample_ticket(),
                message="...",
                prediction=_sample_prediction(),
                model_version="bert@v0.1.0",
            )


# ---------- get_ticket ----------


class TestGetTicket:
    def test_returns_ticket_for_existing_id(self, session_factory: sessionmaker[Session]) -> None:
        ticket = _sample_ticket()
        with session_factory() as session:
            save_ticket(
                session,
                ticket,
                message="...",
                prediction=_sample_prediction(),
                model_version="bert@v0.1.0",
            )

        with session_factory() as session:
            loaded = get_ticket(session, ticket.id)
        assert loaded is not None
        assert isinstance(loaded, Ticket)
        assert loaded.id == ticket.id
        assert loaded.department == ticket.department
        assert loaded.priority == ticket.priority
        assert loaded.sla_hours == ticket.sla_hours
        assert loaded.tags == ticket.tags

    def test_returns_none_for_missing_id(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            assert get_ticket(session, "01HZZZZZZZZZZZZZZZZZZZZZZZ") is None

    def test_round_trip_preserves_routing_fields(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        ticket = _sample_ticket(
            department="Security",
            priority="P1",
            sla_hours=1,
            tags=["card", "security", "fraud"],
        )
        with session_factory() as session:
            save_ticket(
                session,
                ticket,
                message="my card was stolen",
                prediction=_sample_prediction(),
                model_version="bert@v0.1.0",
            )
        with session_factory() as session:
            loaded = get_ticket(session, ticket.id)
        assert loaded is not None
        assert loaded.department == "Security"
        assert loaded.priority == "P1"
        assert loaded.sla_hours == 1
        assert loaded.tags == ["card", "security", "fraud"]

    def test_round_trip_preserves_created_at_utc(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        ticket = _sample_ticket()
        with session_factory() as session:
            save_ticket(
                session,
                ticket,
                message="...",
                prediction=_sample_prediction(),
                model_version="bert@v0.1.0",
            )
        with session_factory() as session:
            loaded = get_ticket(session, ticket.id)
        assert loaded is not None
        assert loaded.created_at.tzinfo is not None
        assert abs((loaded.created_at - ticket.created_at).total_seconds()) < 1.0


# ---------- list_tickets ----------


class TestListTickets:
    def test_empty_when_no_rows(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            assert list_tickets(session) == []

    def test_returns_newest_first(self, session_factory: sessionmaker[Session]) -> None:
        base = datetime.now(UTC)
        ids = [
            "01HZZZZZZZZZZZZZZZZZZZZZZ1",
            "01HZZZZZZZZZZZZZZZZZZZZZZ2",
            "01HZZZZZZZZZZZZZZZZZZZZZZ3",
        ]
        with session_factory() as session:
            for offset, ticket_id in enumerate(ids):
                save_ticket(
                    session,
                    _sample_ticket(
                        ticket_id=ticket_id,
                        created_at=base + timedelta(seconds=offset),
                    ),
                    message=f"msg {offset}",
                    prediction=_sample_prediction(),
                    model_version="bert@v0.1.0",
                )

        with session_factory() as session:
            result = list_tickets(session)
        assert [t.id for t in result] == list(reversed(ids))

    def test_respects_limit(self, session_factory: sessionmaker[Session]) -> None:
        base = datetime.now(UTC)
        with session_factory() as session:
            for i in range(15):
                save_ticket(
                    session,
                    _sample_ticket(
                        ticket_id=f"01HZZZZZZZZZZZZZZZZZZZZZ{i:02d}",
                        created_at=base + timedelta(seconds=i),
                    ),
                    message=f"msg {i}",
                    prediction=_sample_prediction(),
                    model_version="bert@v0.1.0",
                )

        with session_factory() as session:
            result = list_tickets(session, limit=10)
        assert len(result) == 10

    def test_default_limit_is_50(self, session_factory: sessionmaker[Session]) -> None:
        base = datetime.now(UTC)
        with session_factory() as session:
            for i in range(51):
                save_ticket(
                    session,
                    _sample_ticket(
                        ticket_id=f"01HZZZZZZZZZZZZZZZZZZZZZ{i:02d}",
                        created_at=base + timedelta(seconds=i),
                    ),
                    message=f"msg {i}",
                    prediction=_sample_prediction(),
                    model_version="bert@v0.1.0",
                )
        with session_factory() as session:
            result = list_tickets(session)
        assert len(result) == 50

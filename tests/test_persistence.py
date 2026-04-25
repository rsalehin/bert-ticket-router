"""Tests for src.persistence — ORM model, factories, and DB initialization.

CRUD operations (save_ticket, list_tickets, get_ticket) are tested in T-019.
This file covers only the schema, table creation, and engine/session factories.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from src.persistence import Base, TicketRow, init_db, make_engine, make_session

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
        """Calling init_db twice on the same engine must not raise."""
        engine = make_engine("sqlite:///:memory:")
        init_db(engine)
        init_db(engine)  # should be a no-op
        inspector = inspect(engine)
        assert "tickets" in inspector.get_table_names()


# ---------- TicketRow schema ----------


class TestTicketRowColumns:
    """The tickets table has the architecture-section-5 columns and types."""

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
    """Insert and read back a row to sanity-check the ORM mapping."""

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
            assert loaded.created_at.tzinfo is not None  # tz preserved


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
        """Invoke `alembic upgrade head` against the given SQLite path."""
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
        """Schema produced by `alembic upgrade head` matches `init_db()`."""
        # Path A: alembic
        db_alembic = tmp_path / "alembic.db"
        self._run_upgrade(db_alembic)
        eng_a = make_engine(f"sqlite:///{db_alembic}")
        cols_a = {c["name"]: c["nullable"] for c in inspect(eng_a).get_columns("tickets")}
        idx_a = {ix["name"] for ix in inspect(eng_a).get_indexes("tickets")}

        # Path B: init_db
        db_orm = tmp_path / "orm.db"
        eng_b = make_engine(f"sqlite:///{db_orm}")
        init_db(eng_b)
        cols_b = {c["name"]: c["nullable"] for c in inspect(eng_b).get_columns("tickets")}
        idx_b = {ix["name"] for ix in inspect(eng_b).get_indexes("tickets")}

        assert cols_a == cols_b, f"column mismatch: alembic={cols_a}, orm={cols_b}"
        assert idx_a == idx_b, f"index mismatch: alembic={idx_a}, orm={idx_b}"

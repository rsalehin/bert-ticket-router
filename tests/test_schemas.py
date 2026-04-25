"""Tests for src.schemas (Pydantic models for API and domain types)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
    IntentScore,
    Prediction,
    RoutingRule,
    Ticket,
)

# ---------- Fixtures ----------


def _valid_intent_scores() -> list[IntentScore]:
    return [
        IntentScore(intent="card_arrival", confidence=0.94),
        IntentScore(intent="card_delivery_estimate", confidence=0.04),
        IntentScore(intent="lost_or_stolen_card", confidence=0.01),
    ]


def _valid_ticket() -> Ticket:
    return Ticket(
        id="01HXR3F8Z4G2N6P7Q9S0T2U4V6",
        department="Cards",
        priority="P3",
        sla_hours=24,
        tags=["card", "delivery"],
        created_at=datetime(2026, 4, 25, 12, 34, 56, tzinfo=UTC),
    )


# ---------- ClassifyRequest ----------


class TestClassifyRequest:
    def test_valid(self) -> None:
        req = ClassifyRequest(message="My card has not arrived")
        assert req.message == "My card has not arrived"

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            ClassifyRequest(message="")

    def test_too_long_message_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            ClassifyRequest(message="x" * 1001)

    def test_max_length_accepted(self) -> None:
        req = ClassifyRequest(message="x" * 1000)
        assert len(req.message) == 1000


# ---------- IntentScore ----------


class TestIntentScore:
    def test_valid(self) -> None:
        s = IntentScore(intent="card_arrival", confidence=0.5)
        assert s.intent == "card_arrival"
        assert s.confidence == 0.5

    def test_empty_intent_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            IntentScore(intent="", confidence=0.5)

    def test_negative_confidence_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            IntentScore(intent="x", confidence=-0.01)

    def test_above_one_confidence_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            IntentScore(intent="x", confidence=1.01)

    def test_boundaries_accepted(self) -> None:
        IntentScore(intent="x", confidence=0.0)
        IntentScore(intent="x", confidence=1.0)


# ---------- Ticket ----------


class TestTicket:
    def test_valid(self) -> None:
        t = _valid_ticket()
        assert t.priority == "P3"
        assert t.sla_hours == 24

    def test_invalid_priority_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Ticket(
                id="01HXR3F8Z4G2N6P7Q9S0T2U4V6",
                department="Cards",
                priority="P4",  # type: ignore[arg-type]
                sla_hours=24,
                tags=["card"],
                created_at=datetime(2026, 4, 25, 12, 34, 56, tzinfo=UTC),
            )

    def test_zero_sla_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Ticket(
                id="01HXR3F8Z4G2N6P7Q9S0T2U4V6",
                department="Cards",
                priority="P3",
                sla_hours=0,
                tags=["card"],
                created_at=datetime(2026, 4, 25, 12, 34, 56, tzinfo=UTC),
            )

    def test_empty_tags_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Ticket(
                id="01HXR3F8Z4G2N6P7Q9S0T2U4V6",
                department="Cards",
                priority="P3",
                sla_hours=24,
                tags=[],
                created_at=datetime(2026, 4, 25, 12, 34, 56, tzinfo=UTC),
            )

    def test_too_many_tags_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Ticket(
                id="01HXR3F8Z4G2N6P7Q9S0T2U4V6",
                department="Cards",
                priority="P3",
                sla_hours=24,
                tags=[f"tag{i}" for i in range(11)],
                created_at=datetime(2026, 4, 25, 12, 34, 56, tzinfo=UTC),
            )

    def test_id_wrong_length_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Ticket(
                id="too-short",
                department="Cards",
                priority="P3",
                sla_hours=24,
                tags=["card"],
                created_at=datetime(2026, 4, 25, 12, 34, 56, tzinfo=UTC),
            )

    def test_naive_datetime_rejected(self) -> None:
        """created_at must be timezone-aware UTC, not naive."""
        with pytest.raises(PydanticValidationError):
            Ticket(
                id="01HXR3F8Z4G2N6P7Q9S0T2U4V6",
                department="Cards",
                priority="P3",
                sla_hours=24,
                tags=["card"],
                created_at=datetime(2026, 4, 25, 12, 34, 56),  # naive
            )

    def test_serializes_datetime_with_z_suffix(self) -> None:
        t = _valid_ticket()
        dumped = t.model_dump(mode="json")
        assert dumped["created_at"].endswith("Z")
        assert "+00:00" not in dumped["created_at"]


# ---------- ClassifyResponse ----------


class TestClassifyResponse:
    def test_valid(self) -> None:
        r = ClassifyResponse(
            intent="card_arrival",
            confidence=0.94,
            top_k=_valid_intent_scores(),
            ticket=_valid_ticket(),
        )
        assert r.intent == "card_arrival"
        assert len(r.top_k) == 3

    def test_top_k_length_must_be_three(self) -> None:
        with pytest.raises(PydanticValidationError):
            ClassifyResponse(
                intent="card_arrival",
                confidence=0.94,
                top_k=_valid_intent_scores()[:2],  # only 2
                ticket=_valid_ticket(),
            )

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            ClassifyResponse(
                intent="card_arrival",
                confidence=1.5,
                top_k=_valid_intent_scores(),
                ticket=_valid_ticket(),
            )


# ---------- Prediction ----------


class TestPrediction:
    def test_valid(self) -> None:
        p = Prediction(
            intent="card_arrival",
            confidence=0.94,
            top_k=_valid_intent_scores(),
        )
        assert p.intent == "card_arrival"
        assert len(p.top_k) == 3


# ---------- RoutingRule ----------


class TestRoutingRule:
    def test_valid(self) -> None:
        r = RoutingRule(
            intent="card_arrival",
            department="Cards",
            priority="P3",
            sla_hours=24,
            tags=["card", "delivery"],
        )
        assert r.priority == "P3"

    def test_invalid_priority_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            RoutingRule(
                intent="x",
                department="Cards",
                priority="urgent",  # type: ignore[arg-type]
                sla_hours=24,
                tags=["card"],
            )


# ---------- ErrorResponse ----------


class TestErrorResponse:
    def test_valid(self) -> None:
        e = ErrorResponse.from_parts(
            code="VALIDATION_ERROR",
            message="bad input",
            correlation_id="01HXR3F8Z4G2N6P7Q9S0T2U4V6",
        )
        assert e.error.code == "VALIDATION_ERROR"
        assert e.error.message == "bad input"
        assert e.error.correlation_id == "01HXR3F8Z4G2N6P7Q9S0T2U4V6"

    def test_serialized_shape(self) -> None:
        e = ErrorResponse.from_parts(
            code="INTERNAL_ERROR",
            message="boom",
            correlation_id=None,
        )
        dumped = e.model_dump(mode="json")
        assert "error" in dumped
        assert dumped["error"]["code"] == "INTERNAL_ERROR"
        assert dumped["error"]["correlation_id"] is None

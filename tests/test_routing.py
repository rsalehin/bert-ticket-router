"""Tests for src.routing — routing config loader and ticket builder."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from src.errors import IntentNotInRoutingError
from src.errors import ValidationError as AppValidationError
from src.routing import build_ticket, load_routing_rules
from src.schemas import IntentScore, Prediction, RoutingRule, Ticket

# ---------- Helpers ----------


def _write_yaml(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "routing.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


VALID_ENTRY = {
    "intent": "card_arrival",
    "department": "Cards",
    "priority": "P3",
    "sla_hours": 24,
    "tags": ["card", "delivery"],
}


# ---------- load_routing_rules: happy path ----------


class TestLoadRoutingRulesHappyPath:
    def test_returns_dict_keyed_by_intent(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            [
                VALID_ENTRY,
                {**VALID_ENTRY, "intent": "lost_or_stolen_card", "priority": "P1", "sla_hours": 1},
            ],
        )
        rules = load_routing_rules(path)
        assert set(rules.keys()) == {"card_arrival", "lost_or_stolen_card"}

    def test_values_are_routing_rule_instances(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, [VALID_ENTRY])
        rules = load_routing_rules(path)
        rule = rules["card_arrival"]
        assert isinstance(rule, RoutingRule)
        assert rule.department == "Cards"
        assert rule.priority == "P3"
        assert rule.sla_hours == 24
        assert rule.tags == ["card", "delivery"]

    def test_loads_real_repo_routing_yaml(self) -> None:
        rules = load_routing_rules(Path("configs/routing.yaml"))
        assert len(rules) == 77
        assert "card_arrival" in rules
        assert "lost_or_stolen_card" in rules
        assert rules["lost_or_stolen_card"].priority == "P1"


# ---------- load_routing_rules: failure modes ----------


class TestLoadRoutingRulesErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_routing_rules(tmp_path / "does_not_exist.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "routing.yaml"
        path.write_text("not: [valid: yaml: at: all", encoding="utf-8")
        with pytest.raises(AppValidationError):
            load_routing_rules(path)

    def test_top_level_must_be_list(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, {"not": "a list"})
        with pytest.raises(AppValidationError):
            load_routing_rules(path)

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        bad = {**VALID_ENTRY}
        del bad["sla_hours"]
        path = _write_yaml(tmp_path, [bad])
        with pytest.raises(AppValidationError):
            load_routing_rules(path)

    def test_invalid_priority_raises(self, tmp_path: Path) -> None:
        bad = {**VALID_ENTRY, "priority": "URGENT"}
        path = _write_yaml(tmp_path, [bad])
        with pytest.raises(AppValidationError):
            load_routing_rules(path)

    def test_zero_sla_hours_raises(self, tmp_path: Path) -> None:
        bad = {**VALID_ENTRY, "sla_hours": 0}
        path = _write_yaml(tmp_path, [bad])
        with pytest.raises(AppValidationError):
            load_routing_rules(path)

    def test_duplicate_intent_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            [VALID_ENTRY, {**VALID_ENTRY, "department": "Account"}],
        )
        with pytest.raises(AppValidationError):
            load_routing_rules(path)

    def test_error_message_names_offending_intent(self, tmp_path: Path) -> None:
        bad = {**VALID_ENTRY, "intent": "weird_intent", "priority": "URGENT"}
        path = _write_yaml(tmp_path, [bad])
        with pytest.raises(AppValidationError) as info:
            load_routing_rules(path)
        assert "weird_intent" in str(info.value)


# ---------- build_ticket helpers ----------


def _sample_prediction(intent: str = "card_arrival") -> Prediction:
    return Prediction(
        intent=intent,
        confidence=0.9,
        top_k=[
            IntentScore(intent=intent, confidence=0.9),
            IntentScore(intent="card_delivery_estimate", confidence=0.07),
            IntentScore(intent="lost_or_stolen_card", confidence=0.03),
        ],
    )


def _sample_rules() -> dict[str, RoutingRule]:
    return {
        "card_arrival": RoutingRule(
            intent="card_arrival",
            department="Cards",
            priority="P3",
            sla_hours=24,
            tags=["card", "delivery"],
        ),
        "lost_or_stolen_card": RoutingRule(
            intent="lost_or_stolen_card",
            department="Security",
            priority="P1",
            sla_hours=1,
            tags=["card", "security", "fraud"],
        ),
    }


# ---------- build_ticket: happy path ----------


class TestBuildTicketHappyPath:
    def test_returns_ticket_with_routing_fields_populated(self) -> None:
        prediction = _sample_prediction("card_arrival")
        rules = _sample_rules()
        ticket = build_ticket(prediction, rules, model_version="bert@v0.1.0")

        assert isinstance(ticket, Ticket)
        assert ticket.department == "Cards"
        assert ticket.priority == "P3"
        assert ticket.sla_hours == 24
        assert ticket.tags == ["card", "delivery"]

    def test_id_is_26_char_ulid(self) -> None:
        prediction = _sample_prediction("card_arrival")
        rules = _sample_rules()
        ticket = build_ticket(prediction, rules, model_version="bert@v0.1.0")

        assert isinstance(ticket.id, str)
        assert len(ticket.id) == 26
        assert ticket.id.isalnum()
        assert ticket.id.upper() == ticket.id

    def test_created_at_is_utc_and_recent(self) -> None:
        prediction = _sample_prediction("card_arrival")
        rules = _sample_rules()
        before = datetime.now(UTC)
        ticket = build_ticket(prediction, rules, model_version="bert@v0.1.0")
        after = datetime.now(UTC)

        assert ticket.created_at.tzinfo is not None
        assert ticket.created_at.utcoffset() == timedelta(0)
        assert before <= ticket.created_at <= after

    def test_two_calls_produce_distinct_ids(self) -> None:
        prediction = _sample_prediction("card_arrival")
        rules = _sample_rules()
        a = build_ticket(prediction, rules, model_version="bert@v0.1.0")
        b = build_ticket(prediction, rules, model_version="bert@v0.1.0")
        assert a.id != b.id

    def test_routes_security_intent_to_p1(self) -> None:
        prediction = _sample_prediction("lost_or_stolen_card")
        rules = _sample_rules()
        ticket = build_ticket(prediction, rules, model_version="bert@v0.1.0")

        assert ticket.department == "Security"
        assert ticket.priority == "P1"
        assert ticket.sla_hours == 1


# ---------- build_ticket: errors ----------


class TestBuildTicketErrors:
    def test_unknown_intent_raises_intent_not_in_routing(self) -> None:
        prediction = _sample_prediction("not_a_real_intent")
        rules = _sample_rules()
        with pytest.raises(IntentNotInRoutingError) as info:
            build_ticket(prediction, rules, model_version="bert@v0.1.0")
        assert "not_a_real_intent" in str(info.value)

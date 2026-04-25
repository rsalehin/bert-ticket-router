"""Tests for src.routing — routing config loader and ticket builder."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.errors import ValidationError as AppValidationError
from src.routing import load_routing_rules
from src.schemas import RoutingRule

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


# ---------- Happy path ----------


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
        """The committed configs/routing.yaml must load and produce 77 rules."""
        rules = load_routing_rules(Path("configs/routing.yaml"))
        assert len(rules) == 77
        # Spot-check a couple of well-known intents
        assert "card_arrival" in rules
        assert "lost_or_stolen_card" in rules
        assert rules["lost_or_stolen_card"].priority == "P1"


# ---------- Failure modes ----------


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
        # Helpful errors quote the intent name
        assert "weird_intent" in str(info.value)

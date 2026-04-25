"""Tests for configs/routing.yaml — verify the hand-authored routing config.

This is a structural / data-quality test on the YAML file itself. It does not
exercise any code from `src/routing.py`; that module's tests live separately
in `tests/test_routing.py` (T-016 onward).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.data import get_label_names

ROUTING_YAML = Path("configs/routing.yaml")


@pytest.fixture(scope="module")
def routing_data() -> list[dict[str, object]]:
    """Parse `configs/routing.yaml` once per test module."""
    if not ROUTING_YAML.exists():
        pytest.skip(f"{ROUTING_YAML} not found")
    with ROUTING_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, list), "routing.yaml must be a top-level list"
    return data  # type: ignore[no-any-return]


class TestRoutingYamlStructure:
    def test_file_exists(self) -> None:
        assert ROUTING_YAML.exists(), "configs/routing.yaml must exist"

    def test_has_exactly_77_entries(self, routing_data: list[dict[str, object]]) -> None:
        assert len(routing_data) == 77

    def test_every_entry_has_required_fields(self, routing_data: list[dict[str, object]]) -> None:
        required = {"intent", "department", "priority", "sla_hours", "tags"}
        for i, entry in enumerate(routing_data):
            missing = required - set(entry.keys())
            assert not missing, f"entry {i} missing fields: {missing}"

    def test_field_types(self, routing_data: list[dict[str, object]]) -> None:
        for i, entry in enumerate(routing_data):
            assert isinstance(entry["intent"], str), f"entry {i} intent not str"
            assert isinstance(entry["department"], str), f"entry {i} department not str"
            assert isinstance(entry["priority"], str), f"entry {i} priority not str"
            assert isinstance(entry["sla_hours"], int), f"entry {i} sla_hours not int"
            assert isinstance(entry["tags"], list), f"entry {i} tags not list"
            assert all(
                isinstance(t, str) for t in entry["tags"]
            ), f"entry {i} tags must be list[str]"


class TestRoutingYamlValues:
    def test_priorities_in_allowed_set(self, routing_data: list[dict[str, object]]) -> None:
        allowed = {"P1", "P2", "P3"}
        for entry in routing_data:
            assert (
                entry["priority"] in allowed
            ), f"intent {entry['intent']!r} has invalid priority {entry['priority']!r}"

    def test_sla_hours_positive(self, routing_data: list[dict[str, object]]) -> None:
        for entry in routing_data:
            assert entry["sla_hours"] > 0, (  # type: ignore[operator]
                f"intent {entry['intent']!r} has non-positive sla_hours"
            )

    def test_tags_count_between_2_and_4(self, routing_data: list[dict[str, object]]) -> None:
        for entry in routing_data:
            n = len(entry["tags"])  # type: ignore[arg-type]
            assert 2 <= n <= 4, f"intent {entry['intent']!r} has {n} tags; expected 2-4"

    def test_tags_non_empty_strings(self, routing_data: list[dict[str, object]]) -> None:
        for entry in routing_data:
            for tag in entry["tags"]:  # type: ignore[union-attr]
                assert (
                    isinstance(tag, str) and tag.strip()
                ), f"intent {entry['intent']!r} has empty tag"


class TestRoutingYamlCoverage:
    """Every banking77 intent must appear exactly once; no extras."""

    def test_all_intents_covered(self, routing_data: list[dict[str, object]]) -> None:
        yaml_intents = {entry["intent"] for entry in routing_data}
        bank77_intents = set(get_label_names())
        missing = bank77_intents - yaml_intents
        assert not missing, f"missing intents in routing.yaml: {sorted(missing)}"

    def test_no_extra_intents(self, routing_data: list[dict[str, object]]) -> None:
        yaml_intents = {entry["intent"] for entry in routing_data}
        bank77_intents = set(get_label_names())
        extra = yaml_intents - bank77_intents
        assert not extra, f"unknown intents in routing.yaml: {sorted(extra)}"

    def test_no_duplicate_intents(self, routing_data: list[dict[str, object]]) -> None:
        names = [entry["intent"] for entry in routing_data]
        assert len(names) == len(set(names)), "duplicate intents in routing.yaml"


class TestRoutingYamlDepartments:
    """Departments must come from the canonical six-department taxonomy."""

    ALLOWED = {"Cards", "Transfers", "Account", "Security", "Top-ups", "General"}

    def test_department_names_in_taxonomy(self, routing_data: list[dict[str, object]]) -> None:
        for entry in routing_data:
            assert entry["department"] in self.ALLOWED, (
                f"intent {entry['intent']!r} has unknown department " f"{entry['department']!r}"
            )

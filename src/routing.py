"""Routing layer: map predicted intents to structured tickets.

This module owns two responsibilities:

1. **Loading** the hand-authored routing config (`configs/routing.yaml`)
   into a typed, validated, in-memory dict.
2. **Building** a `Ticket` from a `Prediction` and the loaded rules
   (added in T-017).

Validation errors are raised as `src.errors.ValidationError` so the API
layer can map them to a 400 response via the standard exception handler.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError as PydanticValidationError
from ulid import ULID

from src.errors import IntentNotInRoutingError, ValidationError
from src.schemas import Prediction, RoutingRule, Ticket


def load_routing_rules(path: Path | str) -> dict[str, RoutingRule]:
    """Parse and validate `configs/routing.yaml` (or any compatible file).

    Args:
        path: Path to the YAML file. Top-level must be a list of mapping
            entries; each entry must conform to `RoutingRule`.

    Returns:
        Dict keyed by `intent`, value is the validated `RoutingRule`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValidationError: For any of:
            - malformed YAML
            - top-level is not a list
            - an entry fails `RoutingRule` validation
            - duplicate intent names across entries
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Routing config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValidationError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValidationError(f"Routing config must be a top-level list, got {type(data).__name__}")

    rules: dict[str, RoutingRule] = {}
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValidationError(
                f"Routing entry {i} must be a mapping, got {type(entry).__name__}"
            )

        intent_name = entry.get("intent", f"<index {i}>")
        try:
            rule = RoutingRule.model_validate(entry)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"Routing entry {i} (intent={intent_name!r}) failed validation: {exc}"
            ) from exc

        if rule.intent in rules:
            raise ValidationError(f"Duplicate intent in routing config: {rule.intent!r}")

        rules[rule.intent] = rule

    return rules


def build_ticket(
    prediction: Prediction,
    rules: dict[str, RoutingRule],
    model_version: str,
) -> Ticket:
    """Assemble a `Ticket` from a model prediction and the routing rules.

    Args:
        prediction: Output of `Classifier.classify(...)`.
        rules: Map from intent -> RoutingRule, as returned by
            `load_routing_rules`.
        model_version: Audit string identifying the model that produced
            `prediction`. Stored on the persisted DB row by the persistence
            layer; not currently surfaced on the API-facing `Ticket`.

    Returns:
        A populated `Ticket` with a freshly generated ULID id, UTC
        `created_at`, and routing fields copied from the matching rule.

    Raises:
        IntentNotInRoutingError: If `prediction.intent` has no rule in
            `rules`. The error message names the offending intent.
    """
    # `model_version` is accepted for forward compatibility (audit trail at
    # the persistence layer). It is not persisted on the API-facing Ticket.
    _ = model_version

    rule = rules.get(prediction.intent)
    if rule is None:
        raise IntentNotInRoutingError(
            f"No routing rule for predicted intent: {prediction.intent!r}"
        )

    return Ticket(
        id=str(ULID()),
        department=rule.department,
        priority=rule.priority,
        sla_hours=rule.sla_hours,
        tags=list(rule.tags),
        created_at=datetime.now(UTC),
    )

"""Tests for src.data — dataset loader for PolyAI/banking77."""

from __future__ import annotations

import pytest

from src.data import get_label_names, get_num_labels, load_banking77

# ---------- Fast tests (no network) ----------


class TestNumLabels:
    def test_returns_77(self) -> None:
        assert get_num_labels() == 77


class TestLabelNames:
    """`get_label_names` should return the canonical 77 intent strings."""

    def test_returns_77_strings(self) -> None:
        names = get_label_names()
        assert isinstance(names, list)
        assert len(names) == 77
        assert all(isinstance(n, str) for n in names)

    def test_known_intents_present(self) -> None:
        names = get_label_names()
        # Spot-check well-known banking77 intents
        assert "card_arrival" in names
        assert "lost_or_stolen_card" in names
        assert "transfer_not_received_by_recipient" in names

    def test_no_duplicates(self) -> None:
        names = get_label_names()
        assert len(set(names)) == 77

    def test_ordering_is_stable(self) -> None:
        # Two calls must return identical ordered lists
        assert get_label_names() == get_label_names()


# ---------- Slow tests (network + dataset download) ----------


@pytest.mark.slow
class TestLoadBanking77:
    """Live load of PolyAI/banking77 from HuggingFace Datasets."""

    def test_returns_train_and_test_splits(self) -> None:
        ds = load_banking77()
        assert "train" in ds
        assert "test" in ds

    def test_split_sizes(self) -> None:
        ds = load_banking77()
        assert len(ds["train"]) == 10003
        assert len(ds["test"]) == 3080

    def test_row_schema(self) -> None:
        ds = load_banking77()
        row = ds["train"][0]
        assert "text" in row
        assert "label" in row
        assert isinstance(row["text"], str)
        assert isinstance(row["label"], int)
        assert 0 <= row["label"] <= 76

    def test_label_names_match_dataset_features(self) -> None:
        """The hardcoded label list must match the dataset's ClassLabel names."""
        ds = load_banking77()
        ds_names = ds["train"].features["label"].names  # type: ignore[union-attr]
        assert get_label_names() == ds_names

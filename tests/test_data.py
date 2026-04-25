"""Tests for src.data — dataset loader for PolyAI/banking77."""

from __future__ import annotations

from collections import Counter

import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from src.data import get_label_names, get_num_labels, load_banking77, make_splits

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
        assert "card_arrival" in names
        assert "lost_or_stolen_card" in names
        assert "transfer_not_received_by_recipient" in names

    def test_no_duplicates(self) -> None:
        names = get_label_names()
        assert len(set(names)) == 77

    def test_ordering_is_stable(self) -> None:
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
        ds = load_banking77()
        ds_names = ds["train"].features["label"].names  # type: ignore[union-attr]
        assert get_label_names() == ds_names


# ---------- T-008: make_splits ----------


def _toy_dataset(num_classes: int = 5, per_class: int = 100) -> DatasetDict:
    """Build a synthetic DatasetDict with `train` and `test` for fast tests."""
    texts = [f"sample_{i}_class_{c}" for c in range(num_classes) for i in range(per_class)]
    labels = [c for c in range(num_classes) for _ in range(per_class)]
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=[f"intent_{c}" for c in range(num_classes)]),
        }
    )
    train = Dataset.from_dict({"text": texts, "label": labels}, features=features)
    test = Dataset.from_dict({"text": texts[:50], "label": labels[:50]}, features=features)
    return DatasetDict({"train": train, "test": test})


class TestMakeSplitsKeys:
    def test_returns_train_val_test(self) -> None:
        ds = _toy_dataset()
        splits = make_splits(ds, val_ratio=0.1, seed=42)
        assert set(splits.keys()) == {"train", "val", "test"}


class TestMakeSplitsSizes:
    def test_val_is_roughly_ten_percent_of_original_train(self) -> None:
        ds = _toy_dataset(num_classes=5, per_class=100)
        splits = make_splits(ds, val_ratio=0.1, seed=42)
        assert 48 <= len(splits["val"]) <= 52
        assert len(splits["train"]) + len(splits["val"]) == 500

    def test_test_split_is_unchanged(self) -> None:
        ds = _toy_dataset()
        splits = make_splits(ds, val_ratio=0.1, seed=42)
        assert len(splits["test"]) == len(ds["test"])


class TestMakeSplitsStratification:
    def test_every_class_present_in_both_train_and_val(self) -> None:
        ds = _toy_dataset(num_classes=5, per_class=100)
        splits = make_splits(ds, val_ratio=0.1, seed=42)
        train_classes = set(splits["train"]["label"])
        val_classes = set(splits["val"]["label"])
        assert train_classes == set(range(5))
        assert val_classes == set(range(5))

    def test_class_proportions_preserved(self) -> None:
        ds = _toy_dataset(num_classes=5, per_class=100)
        splits = make_splits(ds, val_ratio=0.1, seed=42)
        val_counts = Counter(splits["val"]["label"])
        for c in range(5):
            assert 8 <= val_counts[c] <= 12, f"class {c} has {val_counts[c]} val rows"


class TestMakeSplitsDeterminism:
    def test_same_seed_same_splits(self) -> None:
        ds = _toy_dataset()
        a = make_splits(ds, val_ratio=0.1, seed=42)
        b = make_splits(ds, val_ratio=0.1, seed=42)
        assert a["train"]["text"] == b["train"]["text"]
        assert a["val"]["text"] == b["val"]["text"]

    def test_different_seed_different_splits(self) -> None:
        ds = _toy_dataset()
        a = make_splits(ds, val_ratio=0.1, seed=42)
        b = make_splits(ds, val_ratio=0.1, seed=123)
        assert a["val"]["text"] != b["val"]["text"]


class TestMakeSplitsValRatio:
    def test_invalid_ratio_rejected(self) -> None:
        ds = _toy_dataset()
        with pytest.raises(ValueError):
            make_splits(ds, val_ratio=0.0, seed=42)
        with pytest.raises(ValueError):
            make_splits(ds, val_ratio=1.0, seed=42)
        with pytest.raises(ValueError):
            make_splits(ds, val_ratio=-0.1, seed=42)


@pytest.mark.slow
class TestMakeSplitsOnBanking77:
    def test_all_77_classes_present_in_train_and_val(self) -> None:
        ds = load_banking77()
        splits = make_splits(ds, val_ratio=0.1, seed=42)
        assert set(splits["train"]["label"]) == set(range(77))
        assert set(splits["val"]["label"]) == set(range(77))

    def test_sizes(self) -> None:
        ds = load_banking77()
        splits = make_splits(ds, val_ratio=0.1, seed=42)
        assert len(splits["train"]) + len(splits["val"]) == 10003
        assert 950 <= len(splits["val"]) <= 1050
        assert len(splits["test"]) == 3080

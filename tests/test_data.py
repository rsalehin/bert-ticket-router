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


# ---------- T-009: tokenize_batch ----------


@pytest.mark.slow
class TestTokenizeBatch:
    """Tokenization with real BERT-base-uncased tokenizer."""

    def test_returns_required_keys(self, bert_tokenizer: object) -> None:
        from src.data import tokenize_batch

        out = tokenize_batch(
            {"text": ["hello world"], "label": [0]},
            bert_tokenizer,
            max_len=64,
        )
        assert set(out.keys()) >= {"input_ids", "attention_mask", "label"}

    def test_shape_is_batch_by_max_len(self, bert_tokenizer: object) -> None:
        from src.data import tokenize_batch

        out = tokenize_batch(
            {"text": ["hello", "world", "this is a test"], "label": [0, 1, 2]},
            bert_tokenizer,
            max_len=16,
        )
        assert len(out["input_ids"]) == 3
        for ids in out["input_ids"]:
            assert len(ids) == 16
        for mask in out["attention_mask"]:
            assert len(mask) == 16

    def test_long_input_truncated(self, bert_tokenizer: object) -> None:
        from src.data import tokenize_batch

        long_text = "lorem ipsum " * 200  # ~400 tokens before truncation
        out = tokenize_batch(
            {"text": [long_text], "label": [0]},
            bert_tokenizer,
            max_len=32,
        )
        assert len(out["input_ids"][0]) == 32

    def test_short_input_padded_with_pad_token(self, bert_tokenizer: object) -> None:
        from src.data import tokenize_batch

        pad_id = bert_tokenizer.pad_token_id  # type: ignore[attr-defined]
        out = tokenize_batch(
            {"text": ["hi"], "label": [0]},
            bert_tokenizer,
            max_len=32,
        )
        ids = out["input_ids"][0]
        # First few tokens are real ([CLS] hi [SEP]); the tail must be pad
        assert ids[-1] == pad_id
        assert ids[-2] == pad_id

    def test_attention_mask_marks_padding(self, bert_tokenizer: object) -> None:
        from src.data import tokenize_batch

        out = tokenize_batch(
            {"text": ["hi"], "label": [0]},
            bert_tokenizer,
            max_len=32,
        )
        mask = out["attention_mask"][0]
        # Sum of mask == number of real tokens (small, less than 32)
        real = sum(mask)
        assert 0 < real < 32
        # Trailing positions must all be 0
        assert mask[-1] == 0

    def test_label_passthrough(self, bert_tokenizer: object) -> None:
        from src.data import tokenize_batch

        out = tokenize_batch(
            {"text": ["a", "b", "c"], "label": [7, 42, 76]},
            bert_tokenizer,
            max_len=16,
        )
        assert list(out["label"]) == [7, 42, 76]

    def test_mixed_length_batch(self, bert_tokenizer: object) -> None:
        from src.data import tokenize_batch

        out = tokenize_batch(
            {
                "text": [
                    "hi",
                    "this is a much longer sentence with more tokens than the previous one",
                    "medium",
                ],
                "label": [0, 1, 2],
            },
            bert_tokenizer,
            max_len=24,
        )
        # All output rows must have the same length (max_len)
        assert all(len(ids) == 24 for ids in out["input_ids"])
        assert all(len(m) == 24 for m in out["attention_mask"])

    def test_empty_batch(self, bert_tokenizer: object) -> None:
        from src.data import tokenize_batch

        out = tokenize_batch(
            {"text": [], "label": []},
            bert_tokenizer,
            max_len=16,
        )
        assert out["input_ids"] == []
        assert out["attention_mask"] == []
        assert list(out["label"]) == []

"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@pytest.fixture(scope="session")
def bert_tokenizer() -> Iterator[PreTrainedTokenizerBase]:
    """Session-scoped BERT-base-uncased tokenizer.

    Loads once per test session; cached under ~/.cache/huggingface/hub/ on
    first use. Marked-`slow` tests that need a tokenizer should depend on
    this fixture rather than instantiating their own.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    yield tokenizer


@pytest.fixture(scope="session")
def tiny_bert_name() -> str:
    """Name of the tiny model used to keep tests fast.

    `prajjwal1/bert-tiny` is a 4-layer, 128-hidden BERT (~17 MB) compatible
    with the BERT-base architecture. Suitable for shape and round-trip tests.
    """
    return "prajjwal1/bert-tiny"


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_bert_name: str) -> Iterator[PreTrainedTokenizerBase]:
    """Session-scoped tokenizer for the tiny model."""
    tokenizer = AutoTokenizer.from_pretrained(tiny_bert_name)
    yield tokenizer

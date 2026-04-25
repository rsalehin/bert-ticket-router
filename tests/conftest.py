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

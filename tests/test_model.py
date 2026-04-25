"""Tests for src.model — BertClassifier."""

from __future__ import annotations

import pytest
import torch
from transformers import PreTrainedTokenizerBase

# ---------- Forward pass ----------


@pytest.mark.slow
class TestBertClassifierForward:
    """Forward-pass shape and dtype checks using a tiny BERT."""

    def test_forward_shape(self, tiny_bert_name: str) -> None:
        from src.model import BertClassifier

        model = BertClassifier(base_model_name=tiny_bert_name, num_labels=77)
        input_ids = torch.zeros(2, 64, dtype=torch.long)
        attention_mask = torch.ones(2, 64, dtype=torch.long)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        assert logits.shape == (2, 77)
        assert logits.dtype == torch.float32

    def test_forward_with_real_tokens(
        self,
        tiny_bert_name: str,
        tiny_tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        from src.model import BertClassifier

        model = BertClassifier(base_model_name=tiny_bert_name, num_labels=77)
        encoded = tiny_tokenizer(
            ["hello world", "lost my card"],
            padding="max_length",
            truncation=True,
            max_length=16,
            return_tensors="pt",
        )
        logits = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        assert logits.shape == (2, 77)

    def test_eval_mode_is_deterministic(self, tiny_bert_name: str) -> None:
        from src.model import BertClassifier

        model = BertClassifier(base_model_name=tiny_bert_name, num_labels=77)
        model.eval()
        input_ids = torch.zeros(1, 16, dtype=torch.long)
        attention_mask = torch.ones(1, 16, dtype=torch.long)

        with torch.inference_mode():
            a = model(input_ids=input_ids, attention_mask=attention_mask)
            b = model(input_ids=input_ids, attention_mask=attention_mask)

        assert torch.allclose(a, b)

    def test_dropout_is_active_in_train_mode(self, tiny_bert_name: str) -> None:
        """Two forward passes in train mode should differ due to dropout."""
        from src.model import BertClassifier

        model = BertClassifier(base_model_name=tiny_bert_name, num_labels=77)
        model.train()
        input_ids = torch.zeros(4, 16, dtype=torch.long)
        attention_mask = torch.ones(4, 16, dtype=torch.long)

        torch.manual_seed(0)
        a = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.manual_seed(1)
        b = model(input_ids=input_ids, attention_mask=attention_mask)

        # With dropout active and different seeds, outputs must differ
        assert not torch.allclose(a, b)


# ---------- Save / Load round-trip ----------


@pytest.mark.slow
class TestBertClassifierPersistence:
    """save_pretrained -> from_pretrained must reproduce the model bit-for-bit."""

    def test_save_creates_required_files(
        self,
        tmp_path: object,
        tiny_bert_name: str,
        tiny_tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        from src.model import BertClassifier

        model = BertClassifier(base_model_name=tiny_bert_name, num_labels=77)
        save_dir = tmp_path / "ckpt"  # type: ignore[operator]
        model.save_pretrained(
            save_dir,
            tokenizer=tiny_tokenizer,
            labels=[f"intent_{i}" for i in range(77)],
            manifest={
                "model_version": "tiny@v0.0.1-test",
                "base_model": tiny_bert_name,
                "trained_at": "2026-04-25T12:00:00Z",
                "git_sha": "deadbee",
                "metrics": {"macro_f1": 0.5, "top1_acc": 0.5},
            },
        )

        assert (save_dir / "config.json").exists()
        # Either safetensors or pytorch_model.bin is acceptable; we save safetensors.
        assert (save_dir / "model.safetensors").exists()
        assert (save_dir / "tokenizer_config.json").exists()
        assert (save_dir / "vocab.txt").exists()
        assert (save_dir / "labels.json").exists()
        assert (save_dir / "manifest.json").exists()

    def test_manifest_content(
        self,
        tmp_path: object,
        tiny_bert_name: str,
        tiny_tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        import json as _json

        from src.model import BertClassifier

        model = BertClassifier(base_model_name=tiny_bert_name, num_labels=77)
        save_dir = tmp_path / "ckpt"  # type: ignore[operator]
        manifest_in = {
            "model_version": "tiny@v0.0.1-test",
            "base_model": tiny_bert_name,
            "trained_at": "2026-04-25T12:00:00Z",
            "git_sha": "deadbee",
            "metrics": {"macro_f1": 0.5, "top1_acc": 0.5},
        }
        model.save_pretrained(
            save_dir,
            tokenizer=tiny_tokenizer,
            labels=[f"intent_{i}" for i in range(77)],
            manifest=manifest_in,
        )

        with (save_dir / "manifest.json").open("r", encoding="utf-8") as f:
            manifest_out = _json.load(f)
        assert manifest_out == manifest_in

    def test_labels_content(
        self,
        tmp_path: object,
        tiny_bert_name: str,
        tiny_tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        import json as _json

        from src.model import BertClassifier

        model = BertClassifier(base_model_name=tiny_bert_name, num_labels=77)
        save_dir = tmp_path / "ckpt"  # type: ignore[operator]
        labels_in = [f"intent_{i}" for i in range(77)]
        model.save_pretrained(
            save_dir,
            tokenizer=tiny_tokenizer,
            labels=labels_in,
            manifest={
                "model_version": "x",
                "base_model": tiny_bert_name,
                "trained_at": "2026-04-25T12:00:00Z",
                "git_sha": "x",
                "metrics": {},
            },
        )

        with (save_dir / "labels.json").open("r", encoding="utf-8") as f:
            labels_out = _json.load(f)
        assert labels_out == labels_in

    def test_round_trip_logits_match(
        self,
        tmp_path: object,
        tiny_bert_name: str,
        tiny_tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        from src.model import BertClassifier

        torch.manual_seed(0)
        model = BertClassifier(base_model_name=tiny_bert_name, num_labels=77)
        model.eval()
        save_dir = tmp_path / "ckpt"  # type: ignore[operator]
        model.save_pretrained(
            save_dir,
            tokenizer=tiny_tokenizer,
            labels=[f"intent_{i}" for i in range(77)],
            manifest={
                "model_version": "x",
                "base_model": tiny_bert_name,
                "trained_at": "2026-04-25T12:00:00Z",
                "git_sha": "x",
                "metrics": {},
            },
        )

        loaded = BertClassifier.from_pretrained(save_dir)
        loaded.eval()

        input_ids = torch.zeros(2, 32, dtype=torch.long)
        attention_mask = torch.ones(2, 32, dtype=torch.long)

        with torch.inference_mode():
            a = model(input_ids=input_ids, attention_mask=attention_mask)
            b = loaded(input_ids=input_ids, attention_mask=attention_mask)

        assert torch.allclose(a, b, atol=1e-6)

    def test_load_missing_dir_raises(self, tmp_path: object) -> None:
        from src.model import BertClassifier

        with pytest.raises((FileNotFoundError, OSError)):
            BertClassifier.from_pretrained(tmp_path / "does_not_exist")  # type: ignore[operator]

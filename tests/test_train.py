"""Tests for src.train (training loop and helpers)."""

from __future__ import annotations

import pytest

# ---------- set_seed ----------


class TestSetSeed:
    def test_python_random_is_seeded(self) -> None:
        import random as _random

        from src.train import set_seed

        set_seed(42)
        a = [_random.random() for _ in range(5)]
        set_seed(42)
        b = [_random.random() for _ in range(5)]
        assert a == b

    def test_numpy_is_seeded(self) -> None:
        import numpy as np

        from src.train import set_seed

        set_seed(42)
        a = np.random.randn(5).tolist()
        set_seed(42)
        b = np.random.randn(5).tolist()
        assert a == b

    def test_torch_is_seeded(self) -> None:
        import torch

        from src.train import set_seed

        set_seed(42)
        a = torch.randn(5).tolist()
        set_seed(42)
        b = torch.randn(5).tolist()
        assert a == b

    def test_different_seeds_produce_different_streams(self) -> None:
        import torch

        from src.train import set_seed

        set_seed(42)
        a = torch.randn(5).tolist()
        set_seed(123)
        b = torch.randn(5).tolist()
        assert a != b


# ---------- resolve_device ----------


class TestResolveDevice:
    def test_explicit_cpu(self) -> None:
        from src.train import resolve_device

        assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda_when_unavailable_falls_back_to_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        from src.train import resolve_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        # Even when caller says "cuda", we don't have a GPU here -> cpu
        assert resolve_device("cuda") == "cpu"

    def test_auto_picks_cuda_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        from src.train import resolve_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("auto") == "cuda"

    def test_auto_picks_cpu_when_no_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        from src.train import resolve_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device("auto") == "cpu"


# ---------- build_optimizer_and_scheduler ----------


@pytest.mark.slow
class TestBuildOptimizerAndScheduler:
    """AdamW + linear warmup-then-decay scheduler."""

    def test_optimizer_is_adamw_with_correct_hyperparameters(self, tiny_bert_name: str) -> None:
        import torch

        from src.model import BertClassifier
        from src.train import build_optimizer_and_scheduler

        model = BertClassifier(base_model_name=tiny_bert_name, num_labels=4)
        optimizer, _scheduler = build_optimizer_and_scheduler(
            model,
            learning_rate=2e-5,
            weight_decay=0.01,
            num_training_steps=100,
            warmup_ratio=0.1,
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        # Single param group; HF/PyTorch convention.
        assert len(optimizer.param_groups) == 1
        # The current `lr` reflects the scheduler state (warmup pulls it to 0
        # before any step). Assert against the optimizer's *configured* lr,
        # which is preserved in `defaults`.
        assert optimizer.defaults["lr"] == pytest.approx(2e-5)
        assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)

    def test_scheduler_lr_starts_at_zero(self, tiny_bert_name: str) -> None:
        from src.model import BertClassifier
        from src.train import build_optimizer_and_scheduler

        model = BertClassifier(base_model_name=tiny_bert_name, num_labels=4)
        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            learning_rate=2e-5,
            weight_decay=0.01,
            num_training_steps=100,
            warmup_ratio=0.1,
        )

        # Linear warmup starts at 0 and ramps to base lr.
        # PyTorch reports the *current* lr after step 0 (i.e. before any step).
        assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-9)

    def test_scheduler_lr_peaks_at_end_of_warmup(self, tiny_bert_name: str) -> None:
        from src.model import BertClassifier
        from src.train import build_optimizer_and_scheduler

        model = BertClassifier(base_model_name=tiny_bert_name, num_labels=4)
        num_training_steps = 100
        warmup_ratio = 0.1
        warmup_steps = int(num_training_steps * warmup_ratio)

        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            learning_rate=2e-5,
            weight_decay=0.01,
            num_training_steps=num_training_steps,
            warmup_ratio=warmup_ratio,
        )

        # Step exactly through warmup.
        for _ in range(warmup_steps):
            optimizer.step()
            scheduler.step()

        assert scheduler.get_last_lr()[0] == pytest.approx(2e-5, rel=1e-3)

    def test_scheduler_lr_decays_to_zero_at_end(self, tiny_bert_name: str) -> None:
        from src.model import BertClassifier
        from src.train import build_optimizer_and_scheduler

        model = BertClassifier(base_model_name=tiny_bert_name, num_labels=4)
        num_training_steps = 50
        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            learning_rate=2e-5,
            weight_decay=0.01,
            num_training_steps=num_training_steps,
            warmup_ratio=0.1,
        )

        for _ in range(num_training_steps):
            optimizer.step()
            scheduler.step()

        assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-9)


# ---------- train (end-to-end) ----------


def _toy_train_dataset() -> object:
    """Synthetic 4-class DatasetDict: 32 train / 8 test rows, label-correlated text."""
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

    num_classes = 4
    per_class_train = 8
    per_class_test = 2

    def _build(per_class: int) -> Dataset:
        # Make text correlate with label so a tiny model can pick up *something*.
        texts = [
            f"this is class {c} sample {i}" for c in range(num_classes) for i in range(per_class)
        ]
        labels = [c for c in range(num_classes) for _ in range(per_class)]
        return Dataset.from_dict(
            {"text": texts, "label": labels},
            features=Features(
                {
                    "text": Value("string"),
                    "label": ClassLabel(names=[f"intent_{c}" for c in range(num_classes)]),
                }
            ),
        )

    return DatasetDict({"train": _build(per_class_train), "test": _build(per_class_test)})


@pytest.mark.slow
class TestTrainEndToEnd:
    """Tiny end-to-end run that exercises the full training driver."""

    def test_train_writes_checkpoint_and_report(
        self,
        tmp_path: object,
        tiny_bert_name: str,
    ) -> None:
        from pathlib import Path

        from src.config import Settings
        from src.model import BertClassifier
        from src.train import train

        ckpt_dir = tmp_path / "model"  # type: ignore[operator]
        report_dir = tmp_path / "reports"  # type: ignore[operator]

        settings = Settings(
            base_model_name=tiny_bert_name,
            model_checkpoint_path=ckpt_dir,
            batch_size=4,
            learning_rate=5e-4,  # tiny model -> needs a higher lr to learn anything
            num_epochs=1,
            warmup_ratio=0.1,
            weight_decay=0.0,
            seed=42,
            device="cpu",
            max_len=16,
            top_k=3,
            model_version="tiny@v0.0.1-test",
        )

        result = train(
            settings=settings,
            dataset=_toy_train_dataset(),
            report_dir=report_dir,
        )

        # 1. Returned metrics dict has the expected keys
        assert isinstance(result, dict)
        for k in ("accuracy", "macro_f1", "top1_accuracy"):
            assert k in result

        # 2. Checkpoint files exist
        assert (Path(ckpt_dir) / "config.json").exists()
        assert (Path(ckpt_dir) / "model.safetensors").exists()
        assert (Path(ckpt_dir) / "labels.json").exists()
        assert (Path(ckpt_dir) / "manifest.json").exists()

        # 3. Report files exist
        assert (Path(report_dir) / "metrics.json").exists()
        assert (Path(report_dir) / "confusion_matrix.png").exists()

        # 4. Checkpoint is loadable
        loaded = BertClassifier.from_pretrained(ckpt_dir)
        assert loaded.num_labels == 4

    def test_manifest_records_model_version_from_settings(
        self,
        tmp_path: object,
        tiny_bert_name: str,
    ) -> None:
        import json as _json

        from src.config import Settings
        from src.train import train

        ckpt_dir = tmp_path / "model"  # type: ignore[operator]
        report_dir = tmp_path / "reports"  # type: ignore[operator]

        settings = Settings(
            base_model_name=tiny_bert_name,
            model_checkpoint_path=ckpt_dir,
            batch_size=4,
            num_epochs=1,
            seed=42,
            device="cpu",
            max_len=16,
            model_version="tiny@v0.0.1-foo",
        )
        train(settings=settings, dataset=_toy_train_dataset(), report_dir=report_dir)

        with (ckpt_dir / "manifest.json").open("r", encoding="utf-8") as f:
            manifest = _json.load(f)
        assert manifest["model_version"] == "tiny@v0.0.1-foo"
        assert manifest["base_model"] == tiny_bert_name
        assert "trained_at" in manifest
        assert "metrics" in manifest

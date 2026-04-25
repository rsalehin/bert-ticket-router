"""Custom PyTorch training loop for the BERT ticket router.

Offline training driver. Not imported by the API or any serving code; invoked
from `notebooks/02_train.ipynb` (Colab) or local scripting.

Public surface:
    set_seed(seed)                        : seed Python / NumPy / PyTorch RNGs.
    resolve_device(pref)                  : "cuda" / "cpu" / "auto" -> real device.
    build_optimizer_and_scheduler(...)    : AdamW + linear warmup-then-decay.
    train(settings, dataset=, report_dir=): full pipeline -> final test metrics.
"""

from __future__ import annotations

import os
import random as _random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed: int) -> None:
    """Seed Python `random`, NumPy, and PyTorch (CPU + CUDA) RNGs.

    Also sets `PYTHONHASHSEED` for hash-based determinism across processes
    that may be spawned by data loaders.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str) -> str:
    """Resolve `Settings.device` ("cuda" / "cpu" / "auto") to a real device."""
    pref = pref.lower()
    if pref == "cpu":
        return "cpu"
    if pref in ("cuda", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise ValueError(f"Unknown device preference: {pref!r}")


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    num_training_steps: int,
    warmup_ratio: float,
) -> tuple[AdamW, LambdaLR]:
    """Build AdamW + linear-warmup-then-linear-decay scheduler."""
    from transformers import get_linear_schedule_with_warmup

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def train(
    settings: Any,
    *,
    dataset: Any | None = None,
    report_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run the full training pipeline and return final test-set metrics.

    Args:
        settings: A `src.config.Settings` instance.
        dataset: Optional pre-loaded `DatasetDict` with `train` and `test`.
            Defaults to `load_banking77()`.
        report_dir: Optional output directory for `metrics.json` and
            `confusion_matrix.png`. Defaults to `artifacts/reports`.

    Returns:
        Final test-set metrics dict.

    Side effects:
        - Writes the best-by-val-macro-F1 checkpoint to
          `settings.model_checkpoint_path`.
        - Writes the final test-set report to `report_dir`.
    """
    from transformers import AutoTokenizer

    from src.data import (
        get_label_names,
        load_banking77,
        make_dataloaders,
        make_splits,
    )
    from src.evaluate import evaluate, save_report
    from src.model import BertClassifier

    set_seed(settings.seed)
    device = resolve_device(settings.device)

    if dataset is None:
        dataset = load_banking77()

    splits = make_splits(dataset, val_ratio=0.1, seed=settings.seed)

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name)
    loaders = make_dataloaders(
        splits,
        tokenizer,
        batch_size=settings.batch_size,
        max_len=settings.max_len,
    )

    train_label_feature = splits["train"].features["label"]
    if hasattr(train_label_feature, "names") and train_label_feature.names:
        label_names = list(train_label_feature.names)
    else:
        label_names = get_label_names()
    num_labels = len(label_names)

    model = BertClassifier(
        base_model_name=settings.base_model_name,
        num_labels=num_labels,
    ).to(device)

    num_training_steps = max(1, len(loaders["train"])) * settings.num_epochs
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        num_training_steps=num_training_steps,
        warmup_ratio=settings.warmup_ratio,
    )

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    best_macro_f1 = -1.0
    checkpoint_path = Path(settings.model_checkpoint_path)

    for epoch in range(settings.num_epochs):
        model.train()
        for batch in loaders["train"]:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = torch_functional.cross_entropy(logits, labels)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch_functional.cross_entropy(logits, labels)

            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            scheduler.step()

        val_metrics = evaluate(
            model,
            loaders["val"],
            device=device,
            label_names=label_names,
        )

        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            manifest = {
                "model_version": settings.model_version,
                "base_model": settings.base_model_name,
                "trained_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "git_sha": "unknown",
                "metrics": {
                    "val_macro_f1": float(val_metrics["macro_f1"]),
                    "val_top1_accuracy": float(val_metrics["top1_accuracy"]),
                    "val_top3_accuracy": float(val_metrics["top3_accuracy"]),
                    "epoch": epoch + 1,
                },
            }
            model.save_pretrained(
                checkpoint_path,
                tokenizer=tokenizer,
                labels=label_names,
                manifest=manifest,
            )

    best_model = BertClassifier.from_pretrained(checkpoint_path).to(device)
    test_metrics = evaluate(
        best_model,
        loaders["test"],
        device=device,
        label_names=label_names,
    )

    out_dir = Path(report_dir) if report_dir is not None else Path("artifacts/reports")
    save_report(test_metrics, out_dir)

    return test_metrics

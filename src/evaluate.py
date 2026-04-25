"""Evaluation metrics and report serialization.

`compute_metrics` produces a single dictionary summarizing classifier quality
on a held-out split. `save_report` persists that dictionary as `metrics.json`
plus a `confusion_matrix.png` plot. Both are pure NumPy / scikit-learn /
matplotlib Ã¢â‚¬â€ no model dependency.

The metrics dict has this shape (mostly self-describing):

    {
        "accuracy":         float,                              # equal to top1_accuracy
        "macro_f1":         float,                              # unweighted mean F1
        "top1_accuracy":    float,
        "top3_accuracy":    float,
        "top5_accuracy":    float,
        "per_class": {
            "<label>": {"precision": float, "recall": float, "f1": float, "support": int},
            ...
        },
        "confusion_matrix": list[list[int]],                    # [num_classes, num_classes]
        "labels":           list[str],                          # ordered label names
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

# Ensure a non-interactive backend so headless test/CI environments succeed.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must come after `matplotlib.use`)
import numpy as np
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


def _top_k_accuracy(
    y_true: np.ndarray[Any, np.dtype[Any]],
    y_probs: np.ndarray[Any, np.dtype[Any]],
    k: int,
    num_classes: int,
) -> float:
    """Compute top-k accuracy, capped at `num_classes`."""
    effective_k = min(k, num_classes)
    return float(
        top_k_accuracy_score(
            y_true,
            y_probs,
            k=effective_k,
            labels=np.arange(num_classes),
        )
    )


def compute_metrics(
    y_true: np.ndarray[Any, np.dtype[Any]],
    y_pred: np.ndarray[Any, np.dtype[Any]],
    y_probs: np.ndarray[Any, np.dtype[Any]],
    label_names: list[str],
) -> dict[str, Any]:
    """Compute classification metrics for a held-out split.

    Args:
        y_true: `[N]` int array of true class indices.
        y_pred: `[N]` int array of predicted class indices (argmax of probs).
        y_probs: `[N, num_classes]` float array of class probabilities.
        label_names: Ordered list of class names; `len(label_names) == num_classes`.

    Returns:
        Metrics dict (see module docstring for shape).

    Raises:
        ValueError: If shapes are inconsistent.
    """
    num_classes = len(label_names)
    if y_probs.shape[1] != num_classes:
        raise ValueError(
            f"y_probs has {y_probs.shape[1]} classes but label_names has {num_classes}"
        )
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] != y_probs.shape[0]:
        raise ValueError("y_true, y_pred, and y_probs must have the same length")

    accuracy = float((y_true == y_pred).mean())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        average=None,
        zero_division=0,
    )
    macro_f1 = float(np.mean(f1))

    per_class: dict[str, dict[str, float | int]] = {}
    for i, name in enumerate(label_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = sk_confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "top1_accuracy": _top_k_accuracy(y_true, y_probs, 1, num_classes),
        "top3_accuracy": _top_k_accuracy(y_true, y_probs, 3, num_classes),
        "top5_accuracy": _top_k_accuracy(y_true, y_probs, 5, num_classes),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": list(label_names),
    }


def save_report(metrics: dict[str, Any], out_dir: Path | str) -> None:
    """Persist metrics dict to `metrics.json` and `confusion_matrix.png`.

    Args:
        metrics: Output of `compute_metrics`.
        out_dir: Destination directory; created if missing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. metrics.json
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 2. confusion_matrix.png
    cm = np.asarray(metrics["confusion_matrix"])
    labels: list[str] = metrics["labels"]
    n = len(labels)

    # Auto-size figure so each cell stays roughly square and labels are legible.
    side = max(4.0, 0.3 * n)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6 if n > 20 else 9)
    ax.set_yticklabels(labels, fontsize=6 if n > 20 else 9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate(
    model: Any,
    dataloader: Any,
    device: str,
    label_names: list[str],
) -> dict[str, Any]:
    """Run a full evaluation pass and return metrics.

    Iterates the dataloader once with the model in eval mode under
    `torch.inference_mode()`, collecting predicted classes and class
    probabilities. The model's training-mode flag is restored on return,
    so callers in the middle of a training loop are unaffected.

    Args:
        model: A torch `nn.Module` whose `forward(input_ids, attention_mask)`
            returns logits of shape `[batch, num_labels]`.
        dataloader: A PyTorch `DataLoader` yielding dicts with keys
            `input_ids`, `attention_mask`, `labels`.
        device: Device string accepted by `torch.device(...)`.
        label_names: Ordered list of class names; passed through to
            `compute_metrics`.

    Returns:
        Metrics dict (see `compute_metrics`).
    """
    import torch

    was_training = model.training
    model.eval()
    model.to(device)

    all_preds: list[np.ndarray[Any, np.dtype[Any]]] = []
    all_probs: list[np.ndarray[Any, np.dtype[Any]]] = []
    all_truth: list[np.ndarray[Any, np.dtype[Any]]] = []

    try:
        with torch.inference_mode():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)

                all_preds.append(preds.detach().cpu().numpy())
                all_probs.append(probs.detach().cpu().numpy())
                all_truth.append(labels.detach().cpu().numpy())
    finally:
        if was_training:
            model.train()

    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_truth)

    return compute_metrics(y_true, y_pred, y_probs, label_names)

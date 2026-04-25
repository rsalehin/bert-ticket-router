"""Tests for src.evaluate — metrics computation and report serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluate import compute_metrics, save_report

# ---------- Helpers / fixtures ----------


def _toy_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """4-class hand-crafted example.

    8 samples, 4 classes, 2 misclassifications (samples 6 and 7).
    Probabilities are constructed so the argmax matches `y_pred`.
    """
    label_names = ["alpha", "beta", "gamma", "delta"]

    # Truth and predictions:  6/8 correct = 0.75 accuracy.
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 0, 1, 1, 2, 2, 0, 1])  # last two wrong

    # Probabilities: high mass on the predicted class. Place the true class
    # second-highest for the wrongly predicted samples so they appear in top-2.
    y_probs = np.zeros((8, 4), dtype=float)
    for i, p in enumerate(y_pred):
        y_probs[i, p] = 0.7
    # Put truth as 2nd-best for the misclassified rows
    for i in (6, 7):
        y_probs[i, int(y_true[i])] = 0.25
    # Distribute leftover mass uniformly so each row sums to 1
    for i in range(8):
        remaining = 1.0 - y_probs[i].sum()
        zeros = y_probs[i] == 0
        if zeros.any():
            y_probs[i, zeros] += remaining / zeros.sum()

    return y_true, y_pred, y_probs, label_names


# ---------- compute_metrics ----------


class TestComputeMetricsAccuracy:
    def test_accuracy(self) -> None:
        y_true, y_pred, y_probs, labels = _toy_inputs()
        metrics = compute_metrics(y_true, y_pred, y_probs, labels)
        assert metrics["accuracy"] == pytest.approx(6 / 8)

    def test_top1_equals_accuracy(self) -> None:
        y_true, y_pred, y_probs, labels = _toy_inputs()
        metrics = compute_metrics(y_true, y_pred, y_probs, labels)
        assert metrics["top1_accuracy"] == pytest.approx(metrics["accuracy"])


class TestComputeMetricsTopK:
    def test_top3_accuracy_is_one_when_truth_in_top3(self) -> None:
        """Constructed so the true class is always in the top 2 of the probability vector."""
        y_true, y_pred, y_probs, labels = _toy_inputs()
        metrics = compute_metrics(y_true, y_pred, y_probs, labels)
        assert metrics["top3_accuracy"] == pytest.approx(1.0)

    def test_top5_capped_at_num_classes(self) -> None:
        """With only 4 classes, top-5 is trivially 1.0."""
        y_true, y_pred, y_probs, labels = _toy_inputs()
        metrics = compute_metrics(y_true, y_pred, y_probs, labels)
        assert metrics["top5_accuracy"] == pytest.approx(1.0)


class TestComputeMetricsMacroF1:
    def test_macro_f1_in_unit_interval(self) -> None:
        y_true, y_pred, y_probs, labels = _toy_inputs()
        metrics = compute_metrics(y_true, y_pred, y_probs, labels)
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_macro_f1_perfect_when_all_correct(self) -> None:
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        y_probs = np.eye(4)
        labels = ["a", "b", "c", "d"]
        metrics = compute_metrics(y_true, y_pred, y_probs, labels)
        assert metrics["macro_f1"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)


class TestComputeMetricsPerClass:
    def test_per_class_keys(self) -> None:
        y_true, y_pred, y_probs, labels = _toy_inputs()
        metrics = compute_metrics(y_true, y_pred, y_probs, labels)
        per_class = metrics["per_class"]
        assert set(per_class.keys()) == set(labels)
        for label in labels:
            entry = per_class[label]
            assert set(entry.keys()) == {"precision", "recall", "f1", "support"}
            assert 0.0 <= entry["precision"] <= 1.0
            assert 0.0 <= entry["recall"] <= 1.0
            assert 0.0 <= entry["f1"] <= 1.0
            assert isinstance(entry["support"], int)


class TestComputeMetricsConfusionMatrix:
    def test_shape(self) -> None:
        y_true, y_pred, y_probs, labels = _toy_inputs()
        metrics = compute_metrics(y_true, y_pred, y_probs, labels)
        cm = np.asarray(metrics["confusion_matrix"])
        assert cm.shape == (4, 4)
        assert cm.sum() == len(y_true)

    def test_diagonal_counts_correct_predictions(self) -> None:
        y_true, y_pred, y_probs, labels = _toy_inputs()
        metrics = compute_metrics(y_true, y_pred, y_probs, labels)
        cm = np.asarray(metrics["confusion_matrix"])
        assert int(np.trace(cm)) == int((y_true == y_pred).sum())


class TestComputeMetricsValidation:
    def test_label_count_mismatch_rejected(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        y_probs = np.eye(2)
        with pytest.raises(ValueError):
            compute_metrics(y_true, y_pred, y_probs, ["only_one"])

    def test_probs_shape_mismatch_rejected(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        y_probs = np.zeros((2, 3))  # 3 classes vs 2 labels
        with pytest.raises(ValueError):
            compute_metrics(y_true, y_pred, y_probs, ["a", "b"])


# ---------- save_report ----------


class TestSaveReport:
    def test_writes_metrics_json(self, tmp_path: Path) -> None:
        y_true, y_pred, y_probs, labels = _toy_inputs()
        metrics = compute_metrics(y_true, y_pred, y_probs, labels)
        save_report(metrics, tmp_path)

        path = tmp_path / "metrics.json"
        assert path.exists()
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["accuracy"] == pytest.approx(metrics["accuracy"])
        assert loaded["macro_f1"] == pytest.approx(metrics["macro_f1"])
        assert "per_class" in loaded
        assert "confusion_matrix" in loaded

    def test_writes_confusion_matrix_png(self, tmp_path: Path) -> None:
        y_true, y_pred, y_probs, labels = _toy_inputs()
        metrics = compute_metrics(y_true, y_pred, y_probs, labels)
        save_report(metrics, tmp_path)

        path = tmp_path / "confusion_matrix.png"
        assert path.exists()
        # PNG magic number: first 8 bytes are 89 50 4E 47 0D 0A 1A 0A
        head = path.read_bytes()[:8]
        assert head == b"\x89PNG\r\n\x1a\n"

    def test_creates_dir_if_missing(self, tmp_path: Path) -> None:
        y_true, y_pred, y_probs, labels = _toy_inputs()
        metrics = compute_metrics(y_true, y_pred, y_probs, labels)
        nested = tmp_path / "reports" / "v1"
        save_report(metrics, nested)
        assert (nested / "metrics.json").exists()
        assert (nested / "confusion_matrix.png").exists()

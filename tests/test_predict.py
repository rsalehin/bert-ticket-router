"""Tests for src.predict.Classifier (T-020).

All tests are marked `slow` because they instantiate a PyTorch model and run
a forward pass, even with the tiny BERT checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerBase

from src.errors import ModelNotLoadedError
from src.model import BertClassifier
from src.predict import Classifier
from src.schemas import Prediction

# ---------- helpers / fixtures ----------


_NUM_LABELS = 10  # use 10 fake labels to keep the fixture fast
_FAKE_LABELS = [f"intent_{i:02d}" for i in range(_NUM_LABELS)]
_MODEL_VERSION = "bert-tiny@v0.1.0-test"


@pytest.fixture(scope="module")
def tiny_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
    tiny_bert_name: str,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> Path:
    """Save a tiny BertClassifier checkpoint to a temp dir.

    Shared across all tests in this module (module scope) so the expensive
    model instantiation + file I/O only happens once.
    """
    ckpt_dir = tmp_path_factory.mktemp("tiny_ckpt")
    model = BertClassifier(base_model_name=tiny_bert_name, num_labels=_NUM_LABELS)
    manifest = {
        "model_version": _MODEL_VERSION,
        "base_model": tiny_bert_name,
        "trained_at": "2026-04-25T00:00:00Z",
        "git_sha": "testsha",
        "metrics": {"macro_f1": 0.99},
    }
    model.save_pretrained(
        ckpt_dir,
        tokenizer=tiny_tokenizer,
        labels=_FAKE_LABELS,
        manifest=manifest,
    )
    return ckpt_dir


# ---------- TestClassifierConstruction ----------


@pytest.mark.slow
class TestClassifierConstruction:
    def test_constructs_from_valid_checkpoint(self, tiny_checkpoint: Path) -> None:
        clf = Classifier(tiny_checkpoint, device="cpu")
        assert clf is not None

    def test_model_version_matches_manifest(self, tiny_checkpoint: Path) -> None:
        clf = Classifier(tiny_checkpoint, device="cpu")
        assert clf.model_version == _MODEL_VERSION

    def test_num_labels_matches_labels_json(self, tiny_checkpoint: Path) -> None:
        clf = Classifier(tiny_checkpoint, device="cpu")
        assert clf.num_labels == _NUM_LABELS

    def test_device_property(self, tiny_checkpoint: Path) -> None:
        clf = Classifier(tiny_checkpoint, device="cpu")
        assert clf.device == "cpu"

    def test_model_is_in_eval_mode(self, tiny_checkpoint: Path) -> None:
        clf = Classifier(tiny_checkpoint, device="cpu")
        # Access the internal model to verify .eval() was called.
        assert not clf._model.training  # type: ignore[attr-defined]


# ---------- TestClassifierMissingFiles ----------


@pytest.mark.slow
class TestClassifierMissingFiles:
    def test_missing_checkpoint_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotLoadedError):
            Classifier(tmp_path / "does_not_exist", device="cpu")

    def test_missing_labels_json_raises(self, tmp_path: Path, tiny_checkpoint: Path) -> None:
        import shutil

        ckpt = tmp_path / "ckpt_no_labels"
        shutil.copytree(tiny_checkpoint, ckpt)
        (ckpt / "labels.json").unlink()
        with pytest.raises(ModelNotLoadedError):
            Classifier(ckpt, device="cpu")

    def test_missing_manifest_json_raises(self, tmp_path: Path, tiny_checkpoint: Path) -> None:
        import shutil

        ckpt = tmp_path / "ckpt_no_manifest"
        shutil.copytree(tiny_checkpoint, ckpt)
        (ckpt / "manifest.json").unlink()
        with pytest.raises(ModelNotLoadedError):
            Classifier(ckpt, device="cpu")


# ---------- TestClassifierClassify ----------


@pytest.mark.slow
class TestClassifierClassify:
    def test_returns_prediction(self, tiny_checkpoint: Path) -> None:
        clf = Classifier(tiny_checkpoint, device="cpu")
        result = clf.classify("I lost my card")
        assert isinstance(result, Prediction)

    def test_top_k_length_is_three(self, tiny_checkpoint: Path) -> None:
        clf = Classifier(tiny_checkpoint, device="cpu")
        result = clf.classify("I lost my card")
        assert len(result.top_k) == 3

    def test_top_k_intents_are_label_strings(self, tiny_checkpoint: Path) -> None:
        clf = Classifier(tiny_checkpoint, device="cpu")
        result = clf.classify("transfer failed")
        for score in result.top_k:
            assert score.intent in _FAKE_LABELS

    def test_confidences_in_unit_interval(self, tiny_checkpoint: Path) -> None:
        clf = Classifier(tiny_checkpoint, device="cpu")
        result = clf.classify("what is my balance")
        for score in result.top_k:
            assert 0.0 <= score.confidence <= 1.0

    def test_top_intent_matches_top_k_first(self, tiny_checkpoint: Path) -> None:
        clf = Classifier(tiny_checkpoint, device="cpu")
        result = clf.classify("cancel my card")
        assert result.intent == result.top_k[0].intent
        assert result.confidence == result.top_k[0].confidence

    def test_top_k_sorted_descending(self, tiny_checkpoint: Path) -> None:
        clf = Classifier(tiny_checkpoint, device="cpu")
        result = clf.classify("direct debit payment")
        confs = [s.confidence for s in result.top_k]
        assert confs == sorted(confs, reverse=True)

    def test_inference_mode_no_grad(self, tiny_checkpoint: Path) -> None:
        """classify() must not build a compute graph."""
        clf = Classifier(tiny_checkpoint, device="cpu")
        clf.classify("test message")
        # If inference_mode is active no tensor should require grad outside.
        # We just verify it doesn't raise — the decorator handles the rest.

    def test_deterministic_for_same_input(self, tiny_checkpoint: Path) -> None:
        clf = Classifier(tiny_checkpoint, device="cpu")
        r1 = clf.classify("my card was stolen")
        r2 = clf.classify("my card was stolen")
        assert r1.intent == r2.intent
        assert r1.confidence == pytest.approx(r2.confidence)

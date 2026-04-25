"""Inference pipeline: loads a trained BertClassifier and classifies messages.

`Classifier` is the single entry point for inference. It is constructed once
at server startup, held as a singleton on `app.state`, and shared across all
requests via FastAPI dependency injection.

Key design decisions:
- `.eval()` called at construction: disables dropout for deterministic output.
- `@torch.inference_mode()`: disables autograd tracking, faster than no_grad.
- Labels and model_version loaded from `labels.json` / `manifest.json` written
  by `src/train.py` â€” never inferred at request time.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.errors import ModelNotLoadedError
from src.model import BertClassifier
from src.schemas import IntentScore, Prediction


class Classifier:
    """Loads a trained checkpoint and classifies single messages.

    Args:
        checkpoint_path: Directory produced by `BertClassifier.save_pretrained`.
            Must contain `model.safetensors`, `labels.json`, `manifest.json`,
            and standard HuggingFace tokenizer files.
        device: ``"cpu"``, ``"cuda"``, or ``"auto"`` (uses CUDA when available).
        max_len: Maximum token length for the tokenizer (default 64).

    Raises:
        ModelNotLoadedError: If the checkpoint directory or any required file
            is missing.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        device: str = "auto",
        max_len: int = 64,
    ) -> None:
        ckpt = Path(checkpoint_path)

        if not ckpt.exists():
            raise ModelNotLoadedError(f"Checkpoint directory not found: {ckpt}")

        for required in ("labels.json", "manifest.json"):
            if not (ckpt / required).exists():
                raise ModelNotLoadedError(f"Missing required checkpoint file: {ckpt / required}")

        # Resolve device
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._max_len = max_len

        # Load labels
        with (ckpt / "labels.json").open("r", encoding="utf-8") as f:
            self._labels: list[str] = json.load(f)

        # Load manifest
        with (ckpt / "manifest.json").open("r", encoding="utf-8") as f:
            manifest: dict[str, object] = json.load(f)
        self._model_version = str(manifest["model_version"])

        # Load tokenizer
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(str(ckpt))

        # Load model â€” eval mode, moved to device
        self._model: BertClassifier = BertClassifier.from_pretrained(ckpt).eval().to(self._device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        """Version string from the checkpoint manifest."""
        return self._model_version

    @property
    def num_labels(self) -> int:
        """Number of intent classes."""
        return len(self._labels)

    @property
    def device(self) -> str:
        """Device the model is loaded on."""
        return self._device

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def classify(self, message: str) -> Prediction:
        """Classify a single message and return the top-3 predictions.

        Args:
            message: Raw customer message string.

        Returns:
            A `Prediction` with `intent`, `confidence`, and `top_k` (length 3).
        """
        encoding = self._tokenizer(
            message,
            max_length=self._max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        logits: torch.Tensor = self._model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # [num_labels]

        k = min(3, self.num_labels)
        top_probs, top_indices = torch.topk(probs, k=k)

        top_k = [
            IntentScore(
                intent=self._labels[idx.item()],
                confidence=round(prob.item(), 6),
            )
            for prob, idx in zip(top_probs, top_indices, strict=False)
        ]

        return Prediction(
            intent=top_k[0].intent,
            confidence=top_k[0].confidence,
            top_k=top_k,
        )

"""BertClassifier Ã¢â‚¬â€ fine-tunable BERT with a single classification head.

Architecture: HuggingFace BERT backbone -> Dropout(p=0.1) on the [CLS] pooled
output -> Linear(hidden_size -> num_labels). The forward pass returns raw
logits suitable for cross-entropy loss during training and for softmax /
top-k extraction at inference.

The class is deliberately thin: training, evaluation, and inference logic
live elsewhere (`src/train.py`, `src/evaluate.py`, `src/predict.py`).

On-disk layout produced by `save_pretrained`:

    <dir>/
        config.json              # HF backbone config
        model.safetensors        # backbone + classifier head weights
        tokenizer_config.json    # tokenizer files (saved by tokenizer.save_pretrained)
        vocab.txt
        special_tokens_map.json
        labels.json              # ordered list of intent strings
        manifest.json            # version, base model, training metadata, metrics

`from_pretrained` reverses the process and reconstructs an equivalent module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save
from torch import nn
from transformers import AutoConfig, AutoModel


class BertClassifier(nn.Module):
    """BERT backbone + dropout + linear classification head."""

    def __init__(
        self,
        base_model_name: str = "bert-base-uncased",
        num_labels: int = 77,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.dropout_p = dropout

        config = AutoConfig.from_pretrained(base_model_name)
        self.bert = AutoModel.from_pretrained(base_model_name, config=config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run BERT, take the [CLS] pooled output, classify.

        Returns logits of shape `[batch, num_labels]`.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits: torch.Tensor = self.classifier(pooled)
        return logits

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pretrained(
        self,
        save_dir: Path | str,
        *,
        tokenizer: object,
        labels: list[str],
        manifest: dict[str, Any],
    ) -> None:
        """Persist the backbone, head, tokenizer, label list, and manifest.

        Args:
            save_dir: Destination directory. Created if missing.
            tokenizer: A HuggingFace tokenizer; its `save_pretrained` is invoked.
            labels: Ordered list of `num_labels` intent strings.
            manifest: Free-form dict serialized verbatim to `manifest.json`.

        Raises:
            ValueError: If `len(labels) != self.num_labels`.
        """
        if len(labels) != self.num_labels:
            raise ValueError(f"len(labels)={len(labels)} != num_labels={self.num_labels}")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Backbone config
        self.bert.config.save_pretrained(save_dir)

        # 2. Combined state dict (backbone + classifier head) as safetensors.
        #    HF BERT shares some weight buffers (non-contiguous views); safetensors
        #    refuses non-contiguous tensors, so clone-into-contiguous before saving.
        contiguous_state = {k: v.contiguous().clone() for k, v in self.state_dict().items()}
        safetensors_save(contiguous_state, str(save_dir / "model.safetensors"))

        # 3. Tokenizer files
        tokenizer.save_pretrained(save_dir)  # type: ignore[attr-defined]

        # 4. Labels
        with (save_dir / "labels.json").open("w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

        # 5. Manifest
        with (save_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, save_dir: Path | str) -> BertClassifier:
        """Reconstruct a `BertClassifier` previously saved by `save_pretrained`.

        Reads `manifest.json` for `base_model`, `labels.json` for `num_labels`,
        instantiates a fresh module, then overwrites its weights with the
        contents of `model.safetensors`.

        Raises:
            FileNotFoundError: If `save_dir` or any required file is missing.
        """
        save_dir = Path(save_dir)
        if not save_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")

        manifest_path = save_dir / "manifest.json"
        labels_path = save_dir / "labels.json"
        weights_path = save_dir / "model.safetensors"
        for required in (manifest_path, labels_path, weights_path):
            if not required.exists():
                raise FileNotFoundError(f"Missing required checkpoint file: {required}")

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        with labels_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)

        base_model_name = manifest["base_model"]
        num_labels = len(labels)

        # Build the module shell against the same backbone, then overwrite weights.
        model = cls(base_model_name=base_model_name, num_labels=num_labels)
        state_dict = safetensors_load(str(weights_path))
        model.load_state_dict(state_dict)
        return model

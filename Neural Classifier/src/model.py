"""DistilBertWithFeatures: encoder + numeric features -> MLP head."""
from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

from .config import (
    ENCODER_NAME, HEAD_HIDDEN, HEAD_DROPOUT, NUM_NUMERIC,
    SOURCE_VOCAB, FREEZE_LOWER_LAYERS,
)


class DistilBertWithFeatures(nn.Module):
    def __init__(self, num_labels: int, encoder_name: str = ENCODER_NAME):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(encoder_name)
        hidden = self.distilbert.config.hidden_size
        n_extra = NUM_NUMERIC + len(SOURCE_VOCAB)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_extra, HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(HEAD_HIDDEN, num_labels),
        )
        self.num_labels = num_labels
        self._freeze_lower_layers(FREEZE_LOWER_LAYERS)

    def _freeze_lower_layers(self, n: int) -> None:
        for p in self.distilbert.embeddings.parameters():
            p.requires_grad = False
        for layer in self.distilbert.transformer.layer[:n]:
            for p in layer.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        enc_out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls = enc_out.last_hidden_state[:, 0, :]
        x = torch.cat([cls, numeric_features], dim=-1)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


def save_model(
    model: DistilBertWithFeatures,
    tokenizer: DistilBertTokenizerFast,
    out_dir: Path,
    label_to_id: dict[str, int],
    source_vocab: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.distilbert.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    meta = {
        "num_labels": model.num_labels,
        "label_to_id": label_to_id,
        "id_to_label": {v: k for k, v in label_to_id.items()},
        "source_vocab": source_vocab,
        "encoder_name": ENCODER_NAME,
    }
    (out_dir / "features.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def load_model(out_dir: Path) -> tuple[DistilBertWithFeatures, DistilBertTokenizerFast, dict]:
    meta = json.loads((out_dir / "features.json").read_text())
    tokenizer = DistilBertTokenizerFast.from_pretrained(out_dir)
    model = DistilBertWithFeatures(num_labels=meta["num_labels"], encoder_name=str(out_dir))
    head_state = torch.load(out_dir / "head.pt", map_location="cpu")
    model.head.load_state_dict(head_state)
    model.eval()
    return model, tokenizer, meta

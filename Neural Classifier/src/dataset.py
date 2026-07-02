"""Torch Dataset: tokenize Description + build numeric/source features."""
from __future__ import annotations
import math
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import MAX_LEN, SOURCE_VOCAB, AUG_CHAR_NOISE_RATE


def numeric_features(row) -> np.ndarray:
    """5 numeric features: log_amount, sign, month_sin, month_cos, is_weekend."""
    amount = row.get("Amount")
    try:
        a = float(amount) if amount is not None and not pd.isna(amount) else 0.0
    except (TypeError, ValueError):
        a = 0.0
    log_amount = math.log1p(abs(a))
    sign = 1.0 if a > 0 else (-1.0 if a < 0 else 0.0)

    month = row.get("MONTH")
    try:
        m = int(month) if month is not None and not pd.isna(month) else 0
    except (TypeError, ValueError):
        m = 0
    month_sin = math.sin(2 * math.pi * m / 12) if m else 0.0
    month_cos = math.cos(2 * math.pi * m / 12) if m else 0.0

    date = row.get("DATE")
    is_weekend = 0.0
    try:
        ts = pd.to_datetime(date, errors="coerce")
        if pd.notna(ts):
            is_weekend = 1.0 if ts.weekday() >= 5 else 0.0
    except Exception:
        pass

    return np.array([log_amount, sign, month_sin, month_cos, is_weekend], dtype=np.float32)


def source_onehot(source: str, vocab: list[str] = SOURCE_VOCAB) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    if isinstance(source, str) and source in vocab:
        vec[vocab.index(source)] = 1.0
    return vec


def _char_noise(text: str, rate: float) -> str:
    if not text or rate <= 0:
        return text
    chars = list(text)
    out = []
    for c in chars:
        if random.random() < rate:
            op = random.choice(["swap", "delete", "keep"])
            if op == "delete":
                continue
            if op == "swap" and out:
                out[-1], c = c, out[-1]
        out.append(c)
    return "".join(out)


class TxnDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        label_col: str,
        label_to_id: dict[str, int],
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_col = label_col
        self.label_to_id = label_to_id
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row.get("Description") or "")
        if self.augment:
            text = _char_noise(text, AUG_CHAR_NOISE_RATE)
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        num = numeric_features(row)
        src = source_onehot(str(row.get("Source") or ""))
        features = np.concatenate([num, src]).astype(np.float32)
        label = self.label_to_id[row[self.label_col]]
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "numeric_features": torch.from_numpy(features),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class TxnInferenceDataset(Dataset):
    """No labels — for predict.py."""
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row.get("Description") or "")
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        num = numeric_features(row)
        src = source_onehot(str(row.get("Source") or ""))
        features = np.concatenate([num, src]).astype(np.float32)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "numeric_features": torch.from_numpy(features),
        }

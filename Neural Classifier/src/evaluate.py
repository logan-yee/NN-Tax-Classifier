"""Quick eval on held-out test split — confusion matrix + per-class report."""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import MASTER_LABELED_CSV, STAGE1_DIR, STAGE2_DIR, BATCH_SIZE, SEED
from .dataset import TxnDataset
from .model import load_model
from .train import load_stage_data


def evaluate_stage(stage: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, label_col = load_stage_data(stage)
    out_dir = STAGE1_DIR if stage == 1 else STAGE2_DIR
    model, tokenizer, meta = load_model(out_dir)
    model.to(device).eval()

    label_to_id = meta["label_to_id"]
    id_to_label = {int(k): v for k, v in meta["id_to_label"].items()}
    df = df[df[label_col].isin(label_to_id)].reset_index(drop=True)
    y = df[label_col].map(label_to_id).values

    _, temp_df, _, y_temp = train_test_split(df, y, test_size=0.2, stratify=y, random_state=SEED)
    _, test_df, _, y_test = train_test_split(temp_df, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

    ds = TxnDataset(test_df, tokenizer, label_col, label_to_id, augment=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            p = model(**batch)["logits"].argmax(dim=-1).cpu().tolist()
            preds.extend(p)
            ys.extend(batch["labels"].cpu().tolist())

    y_true_lbl = [id_to_label[i] for i in ys]
    y_pred_lbl = [id_to_label[i] for i in preds]
    print(f"[evaluate stage {stage}] test_size={len(ys)}")
    print(classification_report(y_true_lbl, y_pred_lbl, zero_division=0))
    classes = sorted(set(y_true_lbl) | set(y_pred_lbl))
    cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    args = ap.parse_args()
    evaluate_stage(args.stage)


if __name__ == "__main__":
    main()

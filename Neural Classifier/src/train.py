"""Train Stage-1 (Category) or Stage-2 (Sub-Category1, EXPENSES-only) classifier."""
from __future__ import annotations
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW

from .config import (
    MASTER_LABELED_CSV, STAGE1_DIR, STAGE2_DIR,
    ENCODER_NAME, ENCODER_LR, HEAD_LR, WEIGHT_DECAY, EPOCHS, BATCH_SIZE,
    LABEL_SMOOTHING, EARLY_STOP_PATIENCE, SEED, SOURCE_VOCAB,
)
from .dataset import TxnDataset
from .model import DistilBertWithFeatures, save_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_stage_data(stage: int) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(MASTER_LABELED_CSV)
    if stage == 1:
        label_col = "Category"
        df = df[df[label_col].notna()].reset_index(drop=True)
    elif stage == 2:
        label_col = "Sub-Category1"
        df = df[(df["Category"] == "EXPENSES") & df[label_col].notna()].reset_index(drop=True)
    else:
        raise ValueError(f"unknown stage {stage}")
    return df, label_col


def evaluate(model, loader, device, id_to_label: dict[int, str]) -> dict:
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            p = out["logits"].argmax(dim=-1).cpu().tolist()
            preds.extend(p)
            ys.extend(batch["labels"].cpu().tolist())
    acc = accuracy_score(ys, preds)
    macro = f1_score(ys, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": macro, "y_true": ys, "y_pred": preds}


def train_stage(stage: int, max_rows: int | None = None, epochs: int | None = None) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] stage={stage} device={device}")

    df, label_col = load_stage_data(stage)
    if max_rows:
        df = df.head(max_rows)
    print(f"[train] {len(df)} rows; label_col={label_col}")
    classes = sorted(df[label_col].unique().tolist())
    label_to_id = {c: i for i, c in enumerate(classes)}
    id_to_label = {i: c for c, i in label_to_id.items()}
    print(f"[train] {len(classes)} classes: {classes}")

    y = df[label_col].map(label_to_id).values

    def _try_split(X, y_arr, test_size: float, stratify):
        try:
            return train_test_split(X, y_arr, test_size=test_size, stratify=stratify, random_state=SEED)
        except ValueError as e:
            print(f"[train] stratified split failed ({e}); falling back to random split")
            return train_test_split(X, y_arr, test_size=test_size, stratify=None, random_state=SEED)

    train_df, temp_df, y_train, y_temp = _try_split(df, y, 0.2, y)
    val_df, test_df, y_val, y_test = _try_split(temp_df, y_temp, 0.5, y_temp)
    print(f"[train] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    class_weights_raw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_train)
    # Soften: sqrt of inverse-frequency weights to avoid extreme up-weighting of rare classes
    # (full balanced weights gave REVENUE a ~117x weight, causing severe over-prediction)
    class_weights = np.sqrt(class_weights_raw)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"[train] class_weights (sqrt-softened)={class_weights.round(3).tolist()}")
    print(f"[train] class_weights (raw balanced) ={class_weights_raw.round(3).tolist()}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(ENCODER_NAME)
    train_ds = TxnDataset(train_df, tokenizer, label_col, label_to_id, augment=True)
    val_ds = TxnDataset(val_df, tokenizer, label_col, label_to_id, augment=False)
    test_ds = TxnDataset(test_df, tokenizer, label_col, label_to_id, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = DistilBertWithFeatures(num_labels=len(classes)).to(device)
    encoder_params = [p for n, p in model.distilbert.named_parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    optim = AdamW(
        [
            {"params": encoder_params, "lr": ENCODER_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    n_epochs = epochs or EPOCHS
    total_steps = len(train_loader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=LABEL_SMOOTHING)

    best_macro = -1.0
    patience = 0
    best_state = None
    for epoch in range(n_epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                numeric_features=batch["numeric_features"],
            )["logits"]
            loss = loss_fn(logits, batch["labels"])
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            total += loss.item()
        val_metrics = evaluate(model, val_loader, device, id_to_label)
        print(f"[epoch {epoch+1}/{n_epochs}] train_loss={total/len(train_loader):.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_macro:
            best_macro = val_metrics["macro_f1"]
            best_state = {
                "head": {k: v.cpu().clone() for k, v in model.head.state_dict().items()},
                "encoder": {k: v.cpu().clone() for k, v in model.distilbert.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"[train] early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.head.load_state_dict(best_state["head"])
        model.distilbert.load_state_dict(best_state["encoder"])
        model.to(device)

    test_metrics = evaluate(model, test_loader, device, id_to_label)
    print(f"[test] accuracy={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")
    print(classification_report(
        [id_to_label[i] for i in test_metrics["y_true"]],
        [id_to_label[i] for i in test_metrics["y_pred"]],
        zero_division=0,
    ))

    out_dir = STAGE1_DIR if stage == 1 else STAGE2_DIR
    save_model(model, tokenizer, out_dir, label_to_id, SOURCE_VOCAB)
    print(f"[train] saved -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--max-rows", type=int, default=None, help="limit training rows (smoke test)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    train_stage(args.stage, max_rows=args.max_rows, epochs=args.epochs)


if __name__ == "__main__":
    main()

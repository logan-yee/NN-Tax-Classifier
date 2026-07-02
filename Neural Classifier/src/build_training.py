"""Read MASTER 2025 from Chris&Jen xlsx, normalize labels, write training CSV + label maps."""
from __future__ import annotations
import json
from collections import Counter

import pandas as pd

from .config import (
    MASTER_XLSX, MASTER_SHEET, HEADER_ROW,
    MASTER_LABELED_CSV, LABEL_MAPS_JSON, DATA_DIR,
)
from .schema import MASTER_COLUMNS, conform_to_master
from .label_norm import normalize_labels, filter_for_training, EXCLUDE_CATEGORIES

MIN_SUBCAT_SAMPLES = 5


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[build_training] reading {MASTER_XLSX} sheet={MASTER_SHEET}")
    raw = pd.read_excel(MASTER_XLSX, sheet_name=MASTER_SHEET, header=HEADER_ROW)
    print(f"[build_training] raw shape={raw.shape}")
    df = conform_to_master(raw)
    df = normalize_labels(df)
    before = len(df)
    df = filter_for_training(df)
    print(f"[build_training] after exclude+normalize: {len(df)}/{before} rows")

    cat_counts = Counter(df["Category"].dropna().tolist())
    print(f"[build_training] Category counts: {dict(cat_counts)}")

    expenses = df[df["Category"] == "EXPENSES"]
    sub_counts = Counter(expenses["Sub-Category1"].dropna().tolist())
    print(f"[build_training] Sub-Category1 (EXPENSES only) top 20:")
    for k, v in sub_counts.most_common(20):
        print(f"    {v:5d}  {k}")

    rare_subcats = {k for k, v in sub_counts.items() if v < MIN_SUBCAT_SAMPLES}
    if rare_subcats:
        print(f"[build_training] collapsing {len(rare_subcats)} rare sub-categories (<{MIN_SUBCAT_SAMPLES}) to 'Other'")
        df.loc[
            (df["Category"] == "EXPENSES") & df["Sub-Category1"].isin(rare_subcats),
            "Sub-Category1",
        ] = "Other"

    final_cat = sorted(df["Category"].dropna().unique().tolist())
    final_subcat = sorted(
        df.loc[df["Category"] == "EXPENSES", "Sub-Category1"].dropna().unique().tolist()
    )

    label_maps = {
        "stage1_categories": final_cat,
        "stage2_subcategories": final_subcat,
        "excluded_categories": sorted(EXCLUDE_CATEGORIES),
        "min_subcat_samples": MIN_SUBCAT_SAMPLES,
        "n_train_rows": len(df),
        "n_stage2_rows": int((df["Category"] == "EXPENSES").sum()),
    }
    LABEL_MAPS_JSON.write_text(json.dumps(label_maps, indent=2, ensure_ascii=False))
    print(f"[build_training] wrote {LABEL_MAPS_JSON}")
    print(f"[build_training] stage1 classes: {final_cat}")
    print(f"[build_training] stage2 classes ({len(final_subcat)}): {final_subcat}")

    df.to_csv(MASTER_LABELED_CSV, index=False)
    print(f"[build_training] wrote {len(df)} rows -> {MASTER_LABELED_CSV}")


if __name__ == "__main__":
    main()

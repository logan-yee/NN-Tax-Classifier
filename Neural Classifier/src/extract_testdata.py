"""Drive the Tax Extractor notebook over `testdata_past4months/` for all 5 banks.

Loads the function-definition cell of extractor.ipynb, then calls each
`extract_*_folder_to_csv` with the testdata subfolder + writes CSVs into
`Tax Extractor/output/` (overwriting any existing sample CSVs).
"""
from __future__ import annotations
import json
import os
from pathlib import Path

from .config import REPO_ROOT, TAX_EXTRACTOR_DIR

TESTDATA_DIR = REPO_ROOT / "testdata_past4months"
EXTRACTOR_NB = REPO_ROOT / "Tax Extractor" / "notebooks" / "extractor.ipynb"

BANK_FOLDERS = {
    "BMO_LOC": TESTDATA_DIR / "BMO" / "LOC",
    "BMO_MC": TESTDATA_DIR / "BMO" / "MC",
    "BMO_CK": TESTDATA_DIR / "BMO" / "CK",
    "Tangerine": TESTDATA_DIR / "Tangerin",  # note: testdata uses misspelled folder name
    "TD": TESTDATA_DIR / "TD",
}


def _load_extractor_namespace() -> dict:
    """Execute the first code cell of extractor.ipynb (function definitions)."""
    nb = json.loads(EXTRACTOR_NB.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "extractor_inline", "os": os}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "extract_LOC_MC_transactions_from_pdf" in src:
            exec(src, ns)
            break
    if "extract_LOC_folder_to_csv" not in ns:
        raise RuntimeError("could not locate extract_*_folder_to_csv functions in notebook")
    return ns


def _run_folder(ns: dict, folder: Path, kind: str) -> list[dict]:
    """Per-PDF iteration with None filtering (notebook helpers crash on None)."""
    import glob
    rows: list[dict] = []
    for pdf_file in sorted(glob.glob(str(folder / "*.pdf"))):
        try:
            if kind in ("LOC", "MC"):
                got = ns["extract_LOC_MC_transactions_from_pdf"](pdf_file, kind)
            elif kind in ("CK", "Tangerine"):
                got = ns["extract_CK_Tangerine_transactions_from_pdf"](pdf_file, kind)
            elif kind == "TD":
                got = ns["extract_TD_transactions_from_pdf"](pdf_file)
            else:
                continue
        except Exception as e:
            print(f"  [warn] {Path(pdf_file).name}: {e}")
            continue
        rows.extend([r for r in got if isinstance(r, dict)])
    return rows


def main() -> None:
    import pandas as pd
    TAX_EXTRACTOR_DIR.mkdir(parents=True, exist_ok=True)
    ns = _load_extractor_namespace()

    jobs = [
        ("BMO_LOC", "LOC", "BMO_LOC_transactions.csv", ["TRANS DATE", "POSTING DATE", "DESCRIPTION", "AMOUNT", "CARD TYPE"]),
        ("BMO_MC", "MC", "BMO_MC_transactions.csv", ["TRANS DATE", "POSTING DATE", "DESCRIPTION", "AMOUNT", "CARD TYPE"]),
        ("BMO_CK", "CK", "BMO_CK_transactions.csv", ["TRANS DATE", "DESCRIPTION", "AMOUNT", "CARD TYPE"]),
        ("Tangerine", "Tangerine", "Tangerine_transactions.csv", ["TRANS DATE", "DESCRIPTION", "AMOUNT", "CARD TYPE"]),
        ("TD", "TD", "TD_transactions.csv", ["TRANS DATE", "POSTING DATE", "DESCRIPTION", "AMOUNT", "CARD TYPE"]),
    ]
    for label, kind, out_name, cols in jobs:
        src = BANK_FOLDERS[label]
        if not src.exists():
            print(f"[extract_testdata] SKIP {label}: {src} not found")
            continue
        out = TAX_EXTRACTOR_DIR / out_name
        print(f"[extract_testdata] {label}: {src} -> {out}")
        rows = _run_folder(ns, src, kind)
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out, index=False)
        print(f"  wrote {len(df)} rows")


if __name__ == "__main__":
    main()

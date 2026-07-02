"""Read Tax Extractor and Email Extractor CSVs, map to MASTER 19-col schema."""
from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    TAX_EXTRACTOR_DIR, EMAIL_EXTRACTOR_DIR, INCOMING_CSV, TAX_YEAR, SOURCE_VOCAB,
)
from .schema import MASTER_COLUMNS, empty_master_frame, conform_to_master
from .label_norm import clean_description

ENGLISH_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
FRENCH_MONTHS = {
    "janv": 1, "févr": 2, "fevr": 2, "mars": 3, "avr": 4, "mai": 5,
    "juin": 6, "juil": 7, "août": 8, "aout": 8, "sept": 9,
    "oct": 10, "nov": 11, "déc": 12, "dec": 12,
}
ALL_MONTHS = {**ENGLISH_MONTHS, **FRENCH_MONTHS}

FILENAME_SOURCE_MAP = {
    "BMO_LOC": "BMO_LOC",
    "BMO_MC": "BMO_MC",
    "BMO_CK": "BMO_CK",
    "Tangerine": "Tangerine",
    "TD": "TD_Visa",
}

EMAIL_BANK_TO_SOURCE = {
    "CIBC": "CIBC",
    "BMO": "BMO_CK",
    "TD": "TD_Visa",
    "Tangerine": "Tangerine",
    "RBC": "CIBC",  # no RBC source; map to nearest catch-all is wrong, keep as-is
}


def parse_short_date(s: str, year: int = TAX_YEAR) -> Optional[pd.Timestamp]:
    """Parse 'Mar. 29' / 'Apr 03' / '17mar' / 'mars 29' -> Timestamp."""
    if not isinstance(s, str):
        return None
    raw = s.strip().lower().replace(".", "").replace(",", "")
    if not raw:
        return None
    # Pattern 1: 'mar 29' or 'mars 29' (month first, space, day)
    m = re.match(r"^([a-zàâéèêëîïôûùüç]+)\s+(\d{1,2})$", raw)
    if m:
        mon_token, day = m.group(1), int(m.group(2))
    else:
        # Pattern 2: '17mar' (day first, no space, month abbr)
        m = re.match(r"^(\d{1,2})\s*([a-zàâéèêëîïôûùüç]+)$", raw)
        if m:
            day, mon_token = int(m.group(1)), m.group(2)
        else:
            return None
    mon = ALL_MONTHS.get(mon_token) or ALL_MONTHS.get(mon_token[:4]) or ALL_MONTHS.get(mon_token[:3])
    if mon is None:
        return None
    try:
        return pd.Timestamp(year=year, month=mon, day=day)
    except ValueError:
        return None


def infer_source_from_filename(path: Path) -> str:
    stem = path.stem
    for key, src in FILENAME_SOURCE_MAP.items():
        if stem.startswith(key):
            return src
    return stem


def read_tax_csv(path: Path) -> pd.DataFrame:
    """Tax Extractor CSV (4 or 5 cols) -> MASTER frame."""
    src = infer_source_from_filename(path)
    df = pd.read_csv(path)
    cols = set(df.columns)
    has_posting = "POSTING DATE" in cols
    n = len(df)
    out = empty_master_frame(n)

    out["Source"] = src
    trans_date = df["TRANS DATE"].map(lambda s: parse_short_date(s, TAX_YEAR))
    out["Transaction Date"] = trans_date.values
    out["DATE"] = trans_date.values
    if has_posting:
        out["Posting Date"] = df["POSTING DATE"].map(lambda s: parse_short_date(s, TAX_YEAR)).values
    out["YEAR"] = [d.year if pd.notna(d) else pd.NA for d in trans_date]
    out["MONTH"] = [d.month if pd.notna(d) else pd.NA for d in trans_date]

    out["Description"] = df["DESCRIPTION"].astype(str).map(clean_description).values
    amounts = pd.to_numeric(df["AMOUNT"], errors="coerce")
    card = df["CARD TYPE"].astype(str).str.lower()
    out["Amount"] = amounts.values
    out["Debit"] = np.where(card.eq("debit"), amounts, np.nan)
    out["Credit"] = np.where(card.eq("credit"), amounts, np.nan)
    return out


def _email_direction_to_signed(row) -> tuple[float, float]:
    """Return (debit, credit) for one email row."""
    amount = pd.to_numeric(row.get("TOTAL"), errors="coerce")
    if pd.isna(amount) or amount == 0:
        amount = pd.to_numeric(row.get("AMOUNT"), errors="coerce")
    if pd.isna(amount):
        return (np.nan, np.nan)
    direction = str(row.get("DIRECTION") or "").strip().lower()
    typ = str(row.get("TYPE") or "").strip().lower()
    if direction == "received" or direction == "refund":
        return (np.nan, float(amount))
    if direction == "sent" or direction == "paid":
        return (float(amount), np.nan)
    # invoice/order: no direction, treat as debit (money paid out)
    if typ in ("invoice", "order"):
        return (float(amount), np.nan)
    return (np.nan, np.nan)


def read_email_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    n = len(df)
    out = empty_master_frame(n)

    bank = df.get("BANK", pd.Series([""] * n)).fillna("").astype(str)
    out["Source"] = [EMAIL_BANK_TO_SOURCE.get(b.strip(), b.strip() or "CIBC") for b in bank]

    dates = pd.to_datetime(df["DATE"], errors="coerce")
    out["DATE"] = dates.values
    out["Transaction Date"] = dates.values
    out["YEAR"] = [d.year if pd.notna(d) else pd.NA for d in dates]
    out["MONTH"] = [d.month if pd.notna(d) else pd.NA for d in dates]

    vendor = df.get("VENDOR", pd.Series([""] * n)).fillna("").astype(str)
    contact = df.get("CONTACT", pd.Series([""] * n)).fillna("").astype(str)
    reference = df.get("REFERENCE", pd.Series([""] * n)).fillna("").astype(str)
    subject = df.get("SUBJECT", pd.Series([""] * n)).fillna("").astype(str)
    memo = df.get("MEMO", pd.Series([""] * n)).fillna("").astype(str)

    def build_desc(v, c, r, s, m):
        primary = v.strip() or c.strip()
        secondary = r.strip() or s.strip() or m.strip()
        text = (primary + (" | " + secondary if secondary else "")).strip(" |")
        return clean_description(text)[:256]

    out["Description"] = [
        build_desc(v, c, r, s, m)
        for v, c, r, s, m in zip(vendor, contact, reference, subject, memo)
    ]

    debit_credit = [_email_direction_to_signed(r) for _, r in df.iterrows()]
    out["Debit"] = [d for d, _ in debit_credit]
    out["Credit"] = [c for _, c in debit_credit]
    amount_total = pd.to_numeric(df.get("TOTAL", pd.Series([np.nan] * n)), errors="coerce")
    amount_amt = pd.to_numeric(df.get("AMOUNT", pd.Series([np.nan] * n)), errors="coerce")
    out["Amount"] = amount_total.where(amount_total.notna() & (amount_total != 0), amount_amt).values

    out["GST"] = pd.to_numeric(df.get("GST_HST", pd.Series([np.nan] * n)), errors="coerce").values
    out["QST"] = pd.to_numeric(df.get("QST_PST", pd.Series([np.nan] * n)), errors="coerce").values
    out["Transaction"] = df.get("TYPE", pd.Series([""] * n)).astype(str).values
    out["Memo"] = df.get("MEMO", pd.Series([""] * n)).astype(str).values
    return out


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    key = list(zip(
        df["Source"].astype(str),
        df["DATE"].astype(str),
        df["Amount"].astype(str),
        df["Description"].astype(str).str.slice(0, 80),
    ))
    df = df.assign(_k=key).drop_duplicates("_k").drop(columns=["_k"]).reset_index(drop=True)
    return df


def main(limit: Optional[int] = None) -> Path:
    frames: list[pd.DataFrame] = []
    tax_files = sorted(Path(TAX_EXTRACTOR_DIR).glob("*.csv"))
    for p in tax_files:
        try:
            f = read_tax_csv(p)
            if limit:
                f = f.head(limit)
            frames.append(f)
            print(f"[ingest] tax {p.name}: {len(f)} rows -> source={infer_source_from_filename(p)}")
        except Exception as e:
            print(f"[ingest] FAILED to read {p.name}: {e}")

    email_files = sorted(Path(EMAIL_EXTRACTOR_DIR).glob("email_records_*.csv"))
    for p in email_files:
        try:
            f = read_email_csv(p)
            if limit:
                f = f.head(limit)
            frames.append(f)
            print(f"[ingest] email {p.name}: {len(f)} rows")
        except Exception as e:
            print(f"[ingest] FAILED to read {p.name}: {e}")

    if not frames:
        print("[ingest] no input files found")
        return INCOMING_CSV

    out = pd.concat(frames, ignore_index=True)
    out = conform_to_master(out)
    out = dedupe(out)
    INCOMING_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(INCOMING_CSV, index=False)
    print(f"[ingest] wrote {len(out)} rows to {INCOMING_CSV}")
    return INCOMING_CSV


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="limit rows per file (smoke test)")
    args = ap.parse_args()
    main(limit=args.limit)

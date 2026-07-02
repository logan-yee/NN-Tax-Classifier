"""Predict-time rule layer (ADDITIVE).

Wraps the model output without changing it: if no rule fires, every row ends up
exactly as the two-stage model produced it. Rules apply at predict time only;
nothing here retrains or touches model.py / train.py / dataset.py.

Additions:
  1. Vendor label  -> written into the EXISTING Sub-Category2 column (uppercase).
                      Never touches Category or Sub-Category1.
  2. E-transfer cross-reference: ambiguous Interac/online-transfer rows (no payee
     in the bank line) are matched against email-extractor records by
     date(+/- window) + amount + direction to recover the counterparty CONTACT.
     The contact is classified business/personal via an editable known-contacts
     map + company-suffix heuristic, which yields a Category:
        sending  to business  -> EXPENSES
        receiving from business -> REVENUE
        personal (either way) -> PERSONAL
     The recovered name is written into Note. Transfers that cannot be matched or
     classified are flagged UNRESOLVED (highlighted in a distinct colour).
  3. Category chain (first-match-wins) layered over the model:
        source_td          : TD            -> Category=Advertising,      lean=business
        etransfer_xref      : resolved transfer -> Category from contact
        etransfer_unresolved: transfer, not resolved -> model kept, flagged
        keyword_car         : car-related   -> Category=Business Expense, lean=business
        keyword_facebook    : facebook/meta -> Category=Advertising,      lean=business
        source_prior        : BMO->business / Tangerine->personal lean (Category kept)
        contact_map         : description contact match -> lean (Category kept)
        model               : unchanged fallback
  4. Decision_Source : NEW visible column recording the deciding layer.

Account_Lean and Unresolved are INTERNAL ONLY — computed here to drive rules and
highlighting, but predict.py never writes them to the sheet.
"""
from __future__ import annotations

import re

import pandas as pd

from .config import (
    VENDOR_TERMS, CAR_TERMS, FACEBOOK_TERMS, CONTACT_TERMS,
    TRANSFER_TERMS, COMPANY_SUFFIXES, ETRANSFER_MATCH_WINDOW_DAYS,
)

VENDOR_LABELS = set(VENDOR_TERMS.values())


def _norm(text) -> str:
    """Lowercased string for case-insensitive substring matching."""
    if text is None or (isinstance(text, float) and pd.isna(text)) or pd.isna(text):
        return ""
    return str(text).lower()


def _norm_name(text) -> str:
    """Normalize a contact name: lowercase, collapse whitespace/punctuation."""
    s = _norm(text)
    return re.sub(r"\s{2,}", " ", re.sub(r"[^a-z0-9é èàâêëîïôûùüç]", " ", s)).strip()


def detect_vendor(desc_norm: str) -> str | None:
    """Return the uppercase vendor label if any VENDOR_TERMS substring matches."""
    for term, label in VENDOR_TERMS.items():
        if term in desc_norm:
            return label
    return None


def _match_contact(desc_norm: str) -> str | None:
    """Return 'business' (expense) / 'personal' lean for a description contact match."""
    for term, kind in CONTACT_TERMS.items():
        if term in desc_norm:
            return "business" if kind == "expense" else "personal"
    return None


def _model_lean(model_cat) -> str:
    return "personal" if str(model_cat).upper() in ("PERSONAL", "CAREGIVER") else "business"


# --------------------------------------------------------------------------- #
# E-transfer cross-reference helpers
# --------------------------------------------------------------------------- #

def load_known_contacts(path) -> dict[str, str]:
    """Load the editable contact->lean map (CSV with '#' comment lines)."""
    try:
        df = pd.read_csv(path, comment="#")
    except (FileNotFoundError, OSError, ValueError):
        return {}
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        name = _norm_name(r.get("contact"))
        lean = _norm(r.get("lean")).strip()
        if name and lean in ("business", "personal"):
            out[name] = lean
    return out


def load_etransfer_index(paths) -> pd.DataFrame:
    """Build a normalized index of email-extractor interac records for matching.

    Returns a DataFrame with columns: date, direction (sent|received), amount, contact.
    """
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except (FileNotFoundError, OSError, ValueError):
            continue
        cols = {c.lower(): c for c in df.columns}
        date_c = cols.get("date")
        dir_c = cols.get("direction") or cols.get("transfer_type")
        amt_c = cols.get("amount")
        con_c = cols.get("contact") or cols.get("interactor")
        if not (date_c and dir_c and amt_c and con_c):
            continue
        f = pd.DataFrame({
            "date": pd.to_datetime(df[date_c], errors="coerce", utc=True).dt.tz_localize(None),
            "direction": df[dir_c].astype(str).str.lower().str.strip()
                .replace({"sending": "sent", "receiving": "received"}),
            "amount": pd.to_numeric(df[amt_c], errors="coerce").abs().round(2),
            "contact": df[con_c].astype(str),
        })
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["date", "direction", "amount", "contact"])
    out = pd.concat(frames, ignore_index=True)
    return out[out["amount"].notna()].reset_index(drop=True)


def match_etransfer(index: pd.DataFrame, row_date, amount, direction,
                    window_days: int = ETRANSFER_MATCH_WINDOW_DAYS) -> str | None:
    """Return the best-matching contact name, or None.

    Matches on direction + amount, then nearest date within +/- window_days. If the
    bank date is missing, accepts a unique amount+direction candidate.
    """
    if index is None or len(index) == 0 or direction not in ("sent", "received"):
        return None
    try:
        amt = round(abs(float(amount)), 2)
    except (TypeError, ValueError):
        return None
    cand = index[(index["direction"] == direction) & (index["amount"] == amt)]
    if cand.empty:
        return None
    rdate = pd.to_datetime(row_date, errors="coerce")
    if pd.isna(rdate):
        return str(cand.iloc[0]["contact"]) if len(cand) == 1 else None
    dd = (cand["date"] - rdate).abs().dt.days
    within = cand[dd <= window_days]
    if within.empty:
        return None
    best_i = (within["date"] - rdate).abs().idxmin()
    return str(within.loc[best_i, "contact"])


def contact_lean(contact, known: dict[str, str]) -> str | None:
    """business / personal for a recovered contact, via known map then company suffix."""
    key = _norm_name(contact)
    if not key:
        return None
    for mk, lean in known.items():
        if mk and mk in key:
            return lean
    if any(re.search(rf"\b{re.escape(s)}\b", key) for s in COMPANY_SUFFIXES):
        return "business"
    return None


def etransfer_category(lean, direction) -> str | None:
    if lean == "business":
        return "EXPENSES" if direction == "sent" else "REVENUE"
    if lean == "personal":
        return "PERSONAL"
    return None


def _direction(row) -> str | None:
    if pd.notna(row.get("Debit")):
        return "sent"
    if pd.notna(row.get("Credit")):
        return "received"
    return None


# --------------------------------------------------------------------------- #
# Decision chain
# --------------------------------------------------------------------------- #

def _decide(src, desc_norm, model_cat, row, et_index, known):
    """First-match-wins. Returns (category, lean, decision, unresolved, note_contact)."""
    src = str(src or "")

    # 1. SOURCE (highest): all TD rows -> Advertising / business (absolute).
    if src.startswith("TD"):
        return "Advertising", "business", "source_td", False, None

    # 2. E-TRANSFER cross-reference (ambiguous transfers).
    if et_index is not None and any(t in desc_norm for t in TRANSFER_TERMS):
        direction = _direction(row)
        contact = match_etransfer(et_index, row.get("DATE"), row.get("Amount"), direction)
        if contact is not None:
            lean = contact_lean(contact, known)
            cat = etransfer_category(lean, direction)
            if cat is not None:
                return cat, lean, "etransfer_xref", False, contact
            # contact recovered but not classifiable -> flag for review
            return model_cat, (lean or _model_lean(model_cat)), "etransfer_unresolved", True, contact
        # transfer with no email match -> flag for review
        return model_cat, _model_lean(model_cat), "etransfer_unresolved", True, None

    # 3. KEYWORD: car-related, then facebook/meta.
    if any(t in desc_norm for t in CAR_TERMS):
        return "Business Expense", "business", "keyword_car", False, None
    if any(t in desc_norm for t in FACEBOOK_TERMS):
        return "Advertising", "business", "keyword_facebook", False, None

    # 4. SOURCE PRIOR: BMO -> business lean ; Tangerine -> personal lean.
    if src.startswith("BMO"):
        return model_cat, "business", "source_prior", False, None
    if src == "Tangerine":
        return model_cat, "personal", "source_prior", False, None

    # 5. CONTACT MAP (description-based): only if no higher rule fired.
    cl = _match_contact(desc_norm)
    if cl is not None:
        return model_cat, cl, "contact_map", False, None

    # 6. EXISTING MODEL OUTPUT (unchanged fallback).
    return model_cat, _model_lean(model_cat), "model", False, None


def apply_rules(df: pd.DataFrame, et_index: pd.DataFrame | None = None,
                known_contacts: dict[str, str] | None = None) -> pd.DataFrame:
    """Apply vendor labelling + e-transfer cross-ref + category chain.

    Returns a new frame with Category possibly overridden, Sub-Category2 set for
    vendor rows, Note enriched with recovered e-transfer contacts, and the columns
    Decision_Source plus (internal) Account_Lean / Unresolved added.

    Sub-Category1 is never modified. Rows with no rule match are unchanged.
    et_index / known_contacts default to None, in which case the e-transfer layer
    is skipped and behaviour matches the model-only chain.
    """
    df = df.copy()
    known = known_contacts or {}

    categories, leans, decisions, unresolved = [], [], [], []
    subcat2 = list(df["Sub-Category2"])
    notes = list(df["Note"]) if "Note" in df.columns else [pd.NA] * len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        desc_norm = _norm(row.get("Description"))

        # Vendor label -> EXISTING Sub-Category2 (independent of the category chain).
        label = detect_vendor(desc_norm)
        if label is not None:
            subcat2[i] = label

        cat, lean, decision, unres, note_contact = _decide(
            row.get("Source"), desc_norm, row.get("Category"), row, et_index, known,
        )
        categories.append(cat)
        leans.append(lean)
        decisions.append(decision)
        unresolved.append(unres)

        if note_contact:
            tag = f"xref: {note_contact}"
            existing = notes[i]
            notes[i] = tag if (existing is None or pd.isna(existing) or str(existing).strip() == "") \
                else f"{existing} | {tag}"

    df["Category"] = categories
    df["Sub-Category2"] = subcat2
    df["Note"] = notes
    df["Account_Lean"] = leans          # INTERNAL — excluded from output
    df["Unresolved"] = unresolved       # INTERNAL — drives highlighting
    df["Decision_Source"] = decisions
    return df


def vendor_highlight_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose Sub-Category2 is a vendor label (amber highlight)."""
    return df["Sub-Category2"].isin(VENDOR_LABELS)

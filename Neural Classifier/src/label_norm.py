"""Label normalization: exclusions, category map, sub-category map, description cleaner."""
import re
import pandas as pd

EXCLUDE_CATEGORIES = {
    "TRANSFER", "E-TRANSFER", "BMO_MC pmt", "TD VISA PMT", "OTHER",
    "BORROWING", "BORROWING FROM PERSONAL ACCT.", "BORROW/PAYBACK", "PAY BACK",
    "BOROWING",  # misspelled variant seen in MASTER 2025
}

CATEGORY_MAP = {
    "expenses": "EXPENSES",
    "personal": "PERSONAL",
    "revenue": "REVENUE",
}

SUBCAT_MAP = {
    "car": "Car",
    "entertainement": "Entertainment",
    "entertainment": "Entertainment",
    "office expenses": "Office expenses",
    "in person event expenses": "In Person Event Expenses",
    "groceries": "groceries",
    "restaurant expenses": "Restaurant Expenses",
    "office supplies": "Office Supplies",
    "banking": "banking",
    "travel expenses": "Travel Expenses",
    "shopping": "shopping",
    "paypal": "PAYPAL",
    "bill payment & transfer": "Bill Payment & TRANSFER",
    "revenue": "REVENUE",
    "insurance": "Insurance",
    "business seminar": "Business Seminar",
    "food & clothing & supply": "Food & Clothing & Supply",
    "from loc to ck": "from LOC to CK",
    "office equipment": "Office equipment",
    "trainning": "Training",
    "training": "Training",
    "business trip": "Business Trip",
    "ai startup project": "AI Startup Project",
}


def clean_description(text) -> str:
    """Strip bank prefixes, long reference numbers, normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"^\[.*?\]\s*", "", text)
    text = re.sub(r"\b\d{10,}\b", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _norm_key(val) -> str:
    if not isinstance(val, str):
        return ""
    return val.strip().lower()


def normalize_category(val):
    if not isinstance(val, str):
        return pd.NA
    raw = val.strip()
    if raw == "" or raw == "0":
        return pd.NA
    return CATEGORY_MAP.get(raw.lower(), raw)


def normalize_subcat(val):
    if not isinstance(val, str):
        return pd.NA
    raw = val.strip()
    if raw == "" or raw == "0":
        return pd.NA
    return SUBCAT_MAP.get(raw.lower(), raw)


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """In-place normalization of Category, Sub-Category1, Description columns."""
    df = df.copy()
    df["Category"] = df["Category"].map(normalize_category)
    df["Sub-Category1"] = df["Sub-Category1"].map(normalize_subcat)
    df["Description"] = df["Description"].map(clean_description)
    return df


def filter_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Drop excluded categories, Source==Total rows, and rows missing Category."""
    df = df[df["Source"].astype(str) != "Total"]
    df = df[df["Category"].notna()]
    df = df[~df["Category"].isin(EXCLUDE_CATEGORIES)]
    df = df[df["Description"].astype(str).str.len() > 0]
    return df.reset_index(drop=True)

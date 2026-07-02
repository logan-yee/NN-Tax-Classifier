"""MASTER 19-column schema + empty-row factory."""
import pandas as pd

MASTER_COLUMNS = [
    "Source", "YEAR", "MONTH", "DATE",
    "Transaction Date", "Posting Date",
    "Category", "Sub-Category1", "Sub-Category2",
    "Description", "Amount", "Debit", "Credit", "Balance",
    "Note", "QST", "GST", "Transaction", "Memo",
]

NUMERIC_COLS = {"YEAR", "MONTH", "Amount", "Debit", "Credit", "Balance", "QST", "GST"}


def empty_master_frame(n: int = 0) -> pd.DataFrame:
    df = pd.DataFrame({c: [pd.NA] * n for c in MASTER_COLUMNS})
    return df


def conform_to_master(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex df to MASTER columns, filling missing with NA."""
    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[MASTER_COLUMNS]

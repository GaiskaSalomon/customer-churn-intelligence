"""Load and clean the Telco customer-churn dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn import config


def load_raw(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV. Raises a clear error if the data file is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/get_data.py` first."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean raw data and split into features X and binary target y.

    - Drops the customer id column.
    - `TotalCharges` ships with blank strings -> coerced to numeric (NaN imputed later).
    - Target `Churn` (Yes/No) -> 1/0.
    """
    df = df.copy()
    df = df.drop(columns=[config.ID_COL], errors="ignore")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    y = (df[config.TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=[config.TARGET])
    return X, y


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns)."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical

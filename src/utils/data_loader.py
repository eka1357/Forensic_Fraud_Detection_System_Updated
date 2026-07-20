"""data_loader.py
Utilities for loading, cleaning, and saving the Sparkov fraud-simulation data.

Dataset columns
---------------
Unnamed: 0, trans_date_trans_time, cc_num, merchant, category, amt,
first, last, gender, street, city, state, zip, lat, long, city_pop,
job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud
"""

import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "Datasets"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Columns we keep for modelling (drop PII / free-text that adds noise)
KEEP_COLS = [
    "trans_date_trans_time", "cc_num", "category", "amt",
    "gender", "city", "state", "zip", "lat", "long",
    "city_pop", "unix_time", "merch_lat", "merch_long", "is_fraud",
]


# ── loading ──────────────────────────────────────────────────────────
def load_raw_data(
    train_file: str = "fraudTrain.csv",
    test_file: str = "fraudTest.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw training and test CSV files from the Datasets folder."""
    train_df = pd.read_csv(RAW_DATA_DIR / train_file)
    test_df = pd.read_csv(RAW_DATA_DIR / test_file)
    return train_df, test_df


def load_processed_data(
    train_file: str = "train_processed.csv",
    test_file: str = "test_processed.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load already-cleaned CSV files from data/processed/."""
    train_df = pd.read_csv(PROCESSED_DATA_DIR / train_file)
    test_df = pd.read_csv(PROCESSED_DATA_DIR / test_file)
    return train_df, test_df


# ── cleaning ─────────────────────────────────────────────────────────
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning:
    * Keep only modelling-relevant columns.
    * Drop duplicate rows.
    * Fill missing numeric values with the median.
    * Parse the timestamp column.
    """
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()
    df = df.drop_duplicates().reset_index(drop=True)
    for col in df.select_dtypes(include=["float", "int"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    # Parse timestamp
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(
            df["trans_date_trans_time"], errors="coerce"
        )
    return df


# ── saving ───────────────────────────────────────────────────────────
def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Save a cleaned DataFrame to data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path


# ── convenience ──────────────────────────────────────────────────────
def prepare_data() -> tuple[Path, Path]:
    """Load → clean → save training and test sets. Returns saved paths."""
    train_df, test_df = load_raw_data()
    train_clean = basic_clean(train_df)
    test_clean = basic_clean(test_df)
    train_path = save_processed(train_clean, "train_processed.csv")
    test_path = save_processed(test_clean, "test_processed.csv")
    return train_path, test_path


if __name__ == "__main__":
    tp, tep = prepare_data()
    print(f"Processed training data → {tp}")
    print(f"Processed test data    → {tep}")

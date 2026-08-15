"""data_loader.py — Loading, cleaning, and saving Sparkov fraud-simulation data."""

import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, KEEP_COLS

logger = logging.getLogger(__name__)


def load_raw_data(
    train_file: str = "fraudTrain.csv",
    test_file: str = "fraudTest.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw training and test CSV files from the Datasets folder."""
    train_path = RAW_DATA_DIR / train_file
    test_path = RAW_DATA_DIR / test_file

    if not train_path.exists():
        raise FileNotFoundError(
            f"Raw training file not found at: {train_path}. "
            f"Please place {train_file} in {RAW_DATA_DIR}."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Raw test file not found at: {test_path}. "
            f"Please place {test_file} in {RAW_DATA_DIR}."
        )

    logger.info(f"Loading raw data: {train_path.name} and {test_path.name}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def load_processed_data(
    train_file: str = "train_processed.csv",
    test_file: str = "test_processed.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed CSV files from data/processed/."""
    train_path = PROCESSED_DATA_DIR / train_file
    test_path = PROCESSED_DATA_DIR / test_file

    if not train_path.exists() or not test_path.exists():
        logger.warning(
            f"Processed data not found at {PROCESSED_DATA_DIR}. Attempting to prepare from raw..."
        )
        prepare_data()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning:
    * Keep only modelling-relevant columns.
    * Drop duplicate rows.
    * Fill missing numeric values with median.
    * Parse timestamp column.
    """
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()
    df = df.drop_duplicates().reset_index(drop=True)
    for col in df.select_dtypes(include=["float", "int"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(
            df["trans_date_trans_time"], errors="coerce"
        )
    return df


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Save a cleaned DataFrame to data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    logger.info(f"Saved processed dataset -> {out_path}")
    return out_path


def prepare_data() -> Tuple[Path, Path]:
    """Load → clean → save training and test sets. Returns saved paths."""
    train_df, test_df = load_raw_data()
    train_clean = basic_clean(train_df)
    test_clean = basic_clean(test_df)
    train_path = save_processed(train_clean, "train_processed.csv")
    test_path = save_processed(test_clean, "test_processed.csv")
    return train_path, test_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tp, tep = prepare_data()
    print(f"Processed training data -> {tp}")
    print(f"Processed test data     -> {tep}")

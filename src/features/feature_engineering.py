"""feature_engineering.py
Build the feature matrix used by both Isolation Forest and XGBoost.

Imbalance strategy decision
---------------------------
We use **SMOTE** (Synthetic Minority Over-sampling) at training time rather
than undersampling because the fraud class is < 1 %. Undersampling would
discard too much legitimate-transaction signal; SMOTE generates synthetic
fraud samples so the classifier sees a balanced view without losing data.
Class weights are also set inside XGBoost as a secondary safety net.

Features created
----------------
* ``hour``            – hour of transaction (0-23)
* ``dayofweek``       – day of week (0 = Mon … 6 = Sun)
* ``amt_log``         – log1p of transaction amount (reduces skew)
* ``distance``        – haversine proxy between cardholder and merchant
* ``category_code``   – label-encoded merchant category
* ``benford_score``   – per-category Benford deviation (from Phase 2)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.features.benford import add_benford_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Columns the models will actually use (no label, no raw timestamp)
FEATURE_COLS = [
    "amt", "amt_log", "hour", "dayofweek",
    "lat", "long", "merch_lat", "merch_long",
    "city_pop", "distance", "category_code", "benford_score",
]
LABEL_COL = "is_fraud"


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour and day-of-week from trans_date_trans_time."""
    if "trans_date_trans_time" in df.columns:
        ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df["hour"] = ts.dt.hour.fillna(0).astype(int)
        df["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    elif "unix_time" in df.columns:
        ts = pd.to_datetime(df["unix_time"], unit="s", errors="coerce")
        df["hour"] = ts.dt.hour.fillna(0).astype(int)
        df["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    return df


def _add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform of amount to reduce right-skew."""
    df["amt_log"] = np.log1p(df["amt"])
    return df


def _add_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Simple Euclidean proxy for distance between cardholder and merchant."""
    if {"lat", "long", "merch_lat", "merch_long"}.issubset(df.columns):
        df["distance"] = np.sqrt(
            (df["merch_lat"] - df["lat"]) ** 2
            + (df["merch_long"] - df["long"]) ** 2
        )
    else:
        df["distance"] = 0.0
    return df


def _encode_category(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode the merchant category."""
    if "category" in df.columns:
        df["category_code"] = df["category"].astype("category").cat.codes
    else:
        df["category_code"] = 0
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature-engineering steps and return the enriched DataFrame."""
    df = df.copy()
    df = _add_time_features(df)
    df = _add_amount_features(df)
    df = _add_distance(df)
    df = _encode_category(df)
    df = add_benford_score(df, amount_col="amt", group_col="category")
    return df


def get_feature_matrix(df: pd.DataFrame):
    """Return X (features) and y (label) ready for modelling.

    Missing feature columns are filled with 0 so the pipeline never crashes
    on unseen data that lacks a column.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    for col in FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0.0
    X = X[FEATURE_COLS]          # enforce column order
    X = X.fillna(0)
    y = df[LABEL_COL] if LABEL_COL in df.columns else None
    return X, y


if __name__ == "__main__":
    from src.utils.data_loader import load_processed_data

    train_df, test_df = load_processed_data()
    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)
    X_train, y_train = get_feature_matrix(train_fe)
    X_test, y_test = get_feature_matrix(test_fe)
    print("Training features shape:", X_train.shape)
    print("Test features shape:    ", X_test.shape)
    print("Feature columns:        ", list(X_train.columns))
    if y_train is not None:
        print(f"Train fraud rate: {y_train.mean():.4%}")

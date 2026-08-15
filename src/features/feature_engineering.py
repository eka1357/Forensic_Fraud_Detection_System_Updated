"""feature_engineering.py — Feature matrix construction for Isolation Forest and XGBoost.

Imbalance strategy decision
---------------------------
We use SMOTE (Synthetic Minority Over-sampling) on the training partition rather
than undersampling because the fraud class is < 1%. Undersampling would discard
too much legitimate-transaction signal; SMOTE generates synthetic fraud samples
so the classifier sees a balanced view without losing data. Class weights are
also set inside XGBoost as a secondary safety net.

Features created
----------------
* ``hour``            – hour of transaction (0-23)
* ``dayofweek``       – day of week (0 = Mon … 6 = Sun)
* ``amt_log``         – log1p of transaction amount (reduces right-skew)
* ``distance``        – Haversine distance (km) between cardholder and merchant
* ``category_code``   – deterministic label-encoded merchant category
* ``benford_score``   – per-category Benford deviation
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from src.config import FEATURE_COLS, LABEL_COL
from src.features.benford import add_benford_score

logger = logging.getLogger(__name__)

# Standard categories in the Sparkov fraud simulation dataset
STANDARD_CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
    "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel",
]
DEFAULT_CATEGORY_MAP = {cat: idx for idx, cat in enumerate(sorted(STANDARD_CATEGORIES))}


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour and day-of-week from trans_date_trans_time or unix_time."""
    if "trans_date_trans_time" in df.columns:
        ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df["hour"] = ts.dt.hour.fillna(0).astype(int)
        df["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    elif "unix_time" in df.columns:
        ts = pd.to_datetime(df["unix_time"], unit="s", errors="coerce")
        df["hour"] = ts.dt.hour.fillna(0).astype(int)
        df["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    else:
        df["hour"] = 0
        df["dayofweek"] = 0
    return df


def _add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform of amount (log1p) to reduce right-skew."""
    if "amt" in df.columns:
        df["amt_log"] = np.log1p(np.maximum(df["amt"].fillna(0.0), 0.0))
    else:
        df["amt_log"] = 0.0
    return df


def _haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculate the great-circle distance between two points on Earth in km."""
    # Earth radius in kilometers
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    # Clip to [0, 1] to avoid numerical errors with arcsin/sqrt
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def _add_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate great-circle Haversine distance (km) between cardholder and merchant."""
    required = {"lat", "long", "merch_lat", "merch_long"}
    if required.issubset(df.columns):
        lat1 = df["lat"].fillna(0.0).values
        lon1 = df["long"].fillna(0.0).values
        lat2 = df["merch_lat"].fillna(0.0).values
        lon2 = df["merch_long"].fillna(0.0).values
        df["distance"] = _haversine_distance(lat1, lon1, lat2, lon2)
    else:
        df["distance"] = 0.0
    return df


def _encode_category(
    df: pd.DataFrame, category_map: Optional[Dict[str, int]] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Deterministically encode merchant category to integers using a shared mapping."""
    if category_map is None:
        category_map = DEFAULT_CATEGORY_MAP

    if "category" in df.columns:
        # Map known categories; unseen categories map to len(category_map)
        unknown_code = len(category_map)
        df["category_code"] = df["category"].map(category_map).fillna(unknown_code).astype(int)
    else:
        df["category_code"] = 0

    return df, category_map


def engineer_features(
    df: pd.DataFrame, category_map: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """Apply all feature-engineering steps and return the enriched DataFrame."""
    df = df.copy()
    df = _add_time_features(df)
    df = _add_amount_features(df)
    df = _add_distance(df)
    df, _ = _encode_category(df, category_map=category_map)
    df = add_benford_score(df, amount_col="amt", group_col="category")
    return df


def get_feature_matrix(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Return X (features) and y (label) ready for modelling.

    Missing feature columns are filled with 0 so the pipeline never crashes
    on unseen data that lacks a column.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    for col in FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0.0
    X = X[FEATURE_COLS]  # enforce column order
    X = X.fillna(0.0)
    y = df[LABEL_COL] if LABEL_COL in df.columns else None
    return X, y

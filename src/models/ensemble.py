"""ensemble.py — Combine Isolation Forest + XGBoost + Benford scores into one risk score.

Combination logic
-----------------
We use a weighted average of three normalised (0-1) signals:
  * **XGBoost fraud probability** (weight 0.50) – supervised signal
  * **Isolation Forest anomaly score** (weight 0.30) – unsupervised outlier signal
  * **Benford deviation score** (weight 0.20) – forensic accounting signal

Risk tiers:
  * **Low**    – bottom 90%
  * **Medium** – 90th – 97th percentile
  * **High**   – top 3%
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import joblib
import numpy as np
import pandas as pd
from src.config import (
    ISO_MODEL_PATH,
    XGB_MODEL_PATH,
    W_XGB,
    W_ISO,
    W_BEN,
    TIER_P90,
    TIER_P97,
)

logger = logging.getLogger(__name__)


def _minmax(s: pd.Series) -> pd.Series:
    """Min-max normalise series to [0, 1]. Returns zeros for constant series."""
    lo, hi = s.min(), s.max()
    if np.isnan(lo) or np.isnan(hi) or hi == lo:
        return pd.Series(0.0, index=s.index, dtype=float)
    return ((s - lo) / (hi - lo)).astype(float)


def load_models() -> tuple:
    """Load Isolation Forest and XGBoost model artifacts safely."""
    if not ISO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Isolation Forest model not found at {ISO_MODEL_PATH}. "
            "Please train the model first by running the pipeline."
        )
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found at {XGB_MODEL_PATH}. "
            "Please train the model first by running the pipeline."
        )
    iso_model = joblib.load(ISO_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    return iso_model, xgb_model


def compute_ensemble(
    X: pd.DataFrame,
    iso_model=None,
    xgb_model=None,
) -> pd.DataFrame:
    """Return a DataFrame with ``risk_score`` and ``risk_tier`` columns.

    Parameters
    ----------
    X : feature matrix (must include ``benford_score``).
    iso_model : pre-loaded IsolationForest (loaded from disk if None).
    xgb_model : pre-loaded XGBClassifier (loaded from disk if None).
    """
    if iso_model is None or xgb_model is None:
        loaded_iso, loaded_xgb = load_models()
        iso_model = iso_model or loaded_iso
        xgb_model = xgb_model or loaded_xgb

    # 1. Isolation Forest score (higher = more anomalous)
    iso_raw = pd.Series(-iso_model.decision_function(X), index=X.index)
    iso_norm = _minmax(iso_raw)

    # 2. XGBoost fraud probability
    xgb_prob = pd.Series(xgb_model.predict_proba(X)[:, 1], index=X.index)
    xgb_norm = _minmax(xgb_prob)

    # 3. Benford score
    if "benford_score" in X.columns:
        ben_norm = _minmax(X["benford_score"])
    else:
        ben_norm = pd.Series(0.0, index=X.index)

    # Weighted combination
    combined = W_XGB * xgb_norm + W_ISO * iso_norm + W_BEN * ben_norm

    result = X.copy()
    result["risk_score"] = combined

    # Vectorized Tier Assignment using percentiles
    p90 = float(combined.quantile(TIER_P90))
    p97 = float(combined.quantile(TIER_P97))

    conditions = [
        combined >= p97,
        (combined >= p90) & (combined < p97),
    ]
    choices = ["High", "Medium"]
    result["risk_tier"] = np.select(conditions, choices, default="Low")

    return result


def score_single_transaction(row: dict, iso_model=None, xgb_model=None) -> dict:
    """Score one transaction (dict) and return risk_score + risk_tier."""
    df = pd.DataFrame([row])
    scored = compute_ensemble(df, iso_model, xgb_model)
    return {
        "risk_score": float(scored["risk_score"].iloc[0]),
        "risk_tier": str(scored["risk_tier"].iloc[0]),
    }

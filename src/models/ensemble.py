"""ensemble.py
Combine Isolation Forest + XGBoost + Benford scores into one risk score.

Combination logic
-----------------
We use a weighted average of three normalised (0-1) signals:

  * **XGBoost fraud probability** (weight 0.50) – strongest supervised signal
  * **Isolation Forest anomaly score** (weight 0.30) – unsupervised outlier
  * **Benford deviation score** (weight 0.20) – forensic accounting signal

We weight XGBoost highest because it has seen labels and achieves the best
PR-AUC. Isolation Forest adds value for novel fraud patterns that the
supervised model hasn't learned. Benford catches numeric fabrication.

Risk tiers are defined by percentile thresholds:
  * **Low**    – bottom 90 %
  * **Medium** – 90th – 97th percentile
  * **High**   – top 3 %
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ISO_MODEL_PATH = BASE_DIR / "models" / "isolation_forest.pkl"
XGB_MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud.pkl"

# Weights for the three signals
W_XGB = 0.50
W_ISO = 0.30
W_BEN = 0.20


def _minmax(s: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]. Returns zeros for constant series."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


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
    if iso_model is None:
        iso_model = joblib.load(ISO_MODEL_PATH)
    if xgb_model is None:
        xgb_model = joblib.load(XGB_MODEL_PATH)

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

    # Tier thresholds (percentile-based)
    p90 = combined.quantile(0.90)
    p97 = combined.quantile(0.97)

    def _tier(score):
        if score >= p97:
            return "High"
        elif score >= p90:
            return "Medium"
        else:
            return "Low"

    result["risk_tier"] = combined.apply(_tier)
    return result


def score_single_transaction(row: dict, iso_model=None, xgb_model=None) -> dict:
    """Score one transaction (dict) and return risk_score + risk_tier."""
    df = pd.DataFrame([row])
    scored = compute_ensemble(df, iso_model, xgb_model)
    return {
        "risk_score": float(scored["risk_score"].iloc[0]),
        "risk_tier": scored["risk_tier"].iloc[0],
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from src.utils.data_loader import load_processed_data
    from src.features.feature_engineering import engineer_features, get_feature_matrix

    train_df, _ = load_processed_data()
    train_fe = engineer_features(train_df)
    X, y = get_feature_matrix(train_fe)

    result = compute_ensemble(X)
    print("Risk tier distribution:")
    print(result["risk_tier"].value_counts())
    print("\nSample high-risk transactions:")
    print(result.sort_values("risk_score", ascending=False).head(5)[["risk_score", "risk_tier"]])

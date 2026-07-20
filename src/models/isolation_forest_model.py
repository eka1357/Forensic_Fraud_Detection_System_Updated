"""isolation_forest_model.py
Unsupervised anomaly detection using Isolation Forest.

Contamination parameter choice
------------------------------
We set ``contamination`` to match the observed fraud rate in the training
data (~0.58 %). This means the model expects roughly that fraction of
observations to be anomalous. We avoid the default (0.5) because it would
flag half the data; we also avoid auto because sklearn's heuristic doesn't
account for domain knowledge. Matching the fraud rate provides a
principled starting point that can be refined later.

The Isolation Forest is trained WITHOUT labels — it only sees features.
We evaluate against known labels purely as a sanity check.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"


def train_isolation_forest(
    X: pd.DataFrame,
    contamination: float = 0.006,
    random_state: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest on the feature matrix (no labels).

    Parameters
    ----------
    X : feature matrix
    contamination : expected outlier fraction (default ~0.6 % ≈ fraud rate)
    """
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    joblib.dump(model, MODEL_PATH)
    print(f"  [OK] Isolation Forest saved -> {MODEL_PATH}")
    return model


def get_anomaly_scores(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Return anomaly scores (higher = more anomalous)."""
    return -model.decision_function(X)


def evaluate_against_labels(
    scores: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Sanity-check evaluation using known labels.
    We use ROC-AUC and PR-AUC since thresholds are arbitrary for IF.
    """
    auc_roc = roc_auc_score(y_true, scores)
    auc_pr = average_precision_score(y_true, scores)
    return {"roc_auc": round(auc_roc, 4), "pr_auc": round(auc_pr, 4)}


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from src.utils.data_loader import load_processed_data
    from src.features.feature_engineering import engineer_features, get_feature_matrix

    train_df, _ = load_processed_data()
    train_fe = engineer_features(train_df)
    X, y = get_feature_matrix(train_fe)

    model = train_isolation_forest(X)
    scores = get_anomaly_scores(model, X)

    if y is not None:
        metrics = evaluate_against_labels(scores, y.values)
        print("Isolation Forest sanity-check metrics:", metrics)

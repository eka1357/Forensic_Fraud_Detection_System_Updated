"""isolation_forest_model.py — Unsupervised anomaly detection using Isolation Forest.

Contamination parameter choice
------------------------------
We set ``contamination`` to match the observed fraud rate in the training
data (~0.58%). This means the model expects roughly that fraction of
observations to be anomalous.

The Isolation Forest is trained WITHOUT labels — it only sees features.
We evaluate against known labels purely as a sanity check.
"""

import logging
from pathlib import Path
from typing import Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from src.config import MODELS_DIR, ISO_MODEL_PATH, ISO_CONTAMINATION

logger = logging.getLogger(__name__)


def train_isolation_forest(
    X: pd.DataFrame,
    contamination: float = ISO_CONTAMINATION,
    random_state: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest on the feature matrix without labels."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    joblib.dump(model, ISO_MODEL_PATH)
    logger.info(f"Saved Isolation Forest model -> {ISO_MODEL_PATH}")
    return model


def get_anomaly_scores(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Return anomaly scores where higher values denote greater anomaly."""
    return -model.decision_function(X)


def evaluate_against_labels(
    scores: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, float]:
    """Sanity-check evaluation using known labels."""
    auc_roc = roc_auc_score(y_true, scores)
    auc_pr = average_precision_score(y_true, scores)
    return {"roc_auc": round(float(auc_roc), 4), "pr_auc": round(float(auc_pr), 4)}

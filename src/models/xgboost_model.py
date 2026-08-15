"""xgboost_model.py — Supervised XGBoost classifier for fraud detection.

Imbalance handling
------------------
We use SMOTE on training data to balance minority fraud samples and configure
``scale_pos_weight`` inside XGBoost.

Decision Threshold Optimization
-------------------------------
In severe class imbalance (<1% fraud), the default 0.5 threshold can cause high
false-positive rates (low precision). We optimize the decision threshold on a
validation partition to maximize F1 / achieve a balanced precision-recall trade-off.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from src.config import MODELS_DIR, XGB_MODEL_PATH, METADATA_PATH

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    clf: XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series
) -> float:
    """Find the probability threshold that maximizes F1 score on validation data."""
    probs = clf.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)

    # Avoid division by zero
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )

    best_idx = np.argmax(f1_scores)
    # precision_recall_curve thresholds has len(precisions) - 1
    if best_idx < len(thresholds):
        best_threshold = float(thresholds[best_idx])
    else:
        best_threshold = 0.5

    logger.info(
        f"Optimal threshold found: {best_threshold:.4f} (Val F1: {f1_scores[best_idx]:.4f})"
    )
    return best_threshold


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool = True,
    random_state: int = 42,
) -> XGBClassifier:
    """Train XGBoost with SMOTE oversampling on training data."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if use_smote and y_train.sum() > 5:
        sm = SMOTE(sampling_strategy="auto", random_state=random_state)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        logger.info(f"SMOTE applied: {len(X_train)} -> {len(X_res)} rows")
    else:
        X_res, y_res = X_train, y_train

    pos_weight = (len(y_res) - y_res.sum()) / max(y_res.sum(), 1)

    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=float(pos_weight),
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_res, y_res)
    joblib.dump(clf, XGB_MODEL_PATH)
    logger.info(f"Saved XGBoost model -> {XGB_MODEL_PATH}")
    return clf


def evaluate_model(
    clf: XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> Dict:
    """Compute classification metrics on a held-out set using a specified threshold."""
    probs = clf.predict_proba(X)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    roc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)
    cm = confusion_matrix(y, y_pred)

    metrics = {
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def feature_importance(clf: XGBClassifier, feature_names: list) -> pd.DataFrame:
    """Return a DataFrame of feature importances sorted descending."""
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp

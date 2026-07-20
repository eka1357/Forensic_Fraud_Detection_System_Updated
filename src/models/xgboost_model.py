"""xgboost_model.py
Supervised XGBoost classifier for fraud detection.

Imbalance handling
------------------
We use SMOTE to oversample the minority (fraud) class before training,
AND set ``scale_pos_weight`` inside XGBoost as a secondary guard.
This two-layer approach gives the classifier a balanced view while
preserving all legitimate-transaction signal.

Evaluation philosophy
---------------------
Accuracy is meaningless on < 1 % fraud data (a model that always says
"not fraud" gets 99 %+ accuracy). We focus on:
  - **Precision** – of the ones we flag, how many are real fraud?
  - **Recall** – of the real frauds, how many did we catch?
  - **F1** – harmonic mean balancing the two.
  - **PR-AUC** – precision-recall area under curve (robust to imbalance).

In fraud detection, **recall usually matters more** than precision because
missing a real fraud (false negative) is costlier than investigating a
false alarm (false positive). However, very low precision wastes analyst
time, so we tune for a reasonable balance.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgboost_fraud.pkl"


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    use_smote: bool = True,
    random_state: int = 42,
) -> XGBClassifier:
    """Train XGBoost with SMOTE oversampling.

    Returns the fitted classifier.
    """
    if use_smote and y.sum() > 5:
        sm = SMOTE(sampling_strategy="auto", random_state=random_state)
        X_res, y_res = sm.fit_resample(X, y)
        print(f"  SMOTE: {len(X)} -> {len(X_res)} rows (balanced)")
    else:
        X_res, y_res = X, y

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
    joblib.dump(clf, MODEL_PATH)
    print(f"  [OK] XGBoost saved -> {MODEL_PATH}")
    return clf


def evaluate_model(clf: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute classification metrics on a held-out set."""
    probs = clf.predict_proba(X)[:, 1]
    y_pred = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    roc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)
    cm = confusion_matrix(y, y_pred)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm.tolist(),
    }


def feature_importance(clf: XGBClassifier, feature_names: list) -> pd.DataFrame:
    """Return a DataFrame of feature importances sorted descending."""
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from src.utils.data_loader import load_processed_data
    from src.features.feature_engineering import engineer_features, get_feature_matrix

    train_df, test_df = load_processed_data()
    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)
    X_train, y_train = get_feature_matrix(train_fe)
    X_test, y_test = get_feature_matrix(test_fe)

    clf = train_xgboost(X_train, y_train)
    metrics = evaluate_model(clf, X_test, y_test)
    print("\nXGBoost test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    imp = feature_importance(clf, list(X_train.columns))
    print("\nTop 5 features:")
    print(imp.head())

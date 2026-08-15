"""run_pipeline.py — Master runner for the Forensic Fraud Detection Pipeline.

Executes:
  Phase 1: Data loading & cleaning
  Phase 2: Benford's Law module sanity checks
  Phase 3: Feature engineering with consistent categorical mapping
  Phase 4: Isolation Forest unsupervised training
  Phase 5: XGBoost training (SMOTE on train, threshold calibration on validation, final test eval)
  Phase 6: Multi-signal ensemble risk scoring
  Phase 7: UK regulatory compliance flag mapping
"""

import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    REPORTS_DIR,
    MODELS_DIR,
    METADATA_PATH,
    ISO_CONTAMINATION,
    LABEL_COL,
)
from src.utils.data_loader import prepare_data, load_processed_data
from src.utils.eda import generate_eda_report
from src.features.benford import benford_chi2
from src.features.feature_engineering import engineer_features, get_feature_matrix
from src.models.isolation_forest_model import (
    train_isolation_forest,
    get_anomaly_scores,
    evaluate_against_labels,
)
from src.models.xgboost_model import (
    train_xgboost,
    find_optimal_threshold,
    evaluate_model,
    feature_importance,
)
from src.models.ensemble import compute_ensemble
from src.compliance.compliance_engine import get_compliance_flags, format_flags_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def phase1():
    """Phase 1: Data Understanding & Cleaning."""
    logger.info("=" * 60)
    logger.info("Phase 1 — Data Understanding & Cleaning")
    logger.info("=" * 60)

    train_path, test_path = prepare_data()
    logger.info(f"Processed train saved -> {train_path}")
    logger.info(f"Processed test saved  -> {test_path}")

    train_df, test_df = load_processed_data()
    generate_eda_report(train_df, test_df)
    fraud_rate = train_df["is_fraud"].mean()
    logger.info(f"Training fraud rate: {fraud_rate:.4%}")
    return train_df, test_df


def phase2():
    """Phase 2: Benford's Law Module."""
    logger.info("=" * 60)
    logger.info("Phase 2 — Benford's Law Module")
    logger.info("=" * 60)

    sample = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9] * 100)
    score = benford_chi2(sample)
    logger.info(f"Benford module initialized. Uniform digits chi2: {score:.1f}")


def phase3(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Phase 3: Feature Engineering with Train/Validation partition."""
    logger.info("=" * 60)
    logger.info("Phase 3 — Feature Engineering")
    logger.info("=" * 60)

    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)

    X_train_full, y_train_full = get_feature_matrix(train_fe)
    X_test, y_test = get_feature_matrix(test_fe)

    # Create an 80/20 train/validation split for threshold tuning and hyperparameter checks
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )

    logger.info(f"Train split:      {X_train.shape} (Fraud: {y_train.mean():.4%})")
    logger.info(f"Validation split: {X_val.shape} (Fraud: {y_val.mean():.4%})")
    logger.info(f"Test split:       {X_test.shape} (Fraud: {y_test.mean():.4%})")
    logger.info(f"Feature columns:  {list(X_train.columns)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def phase4(X_train: pd.DataFrame, y_train: pd.Series):
    """Phase 4: Isolation Forest (Unsupervised)."""
    logger.info("=" * 60)
    logger.info("Phase 4 — Isolation Forest (Unsupervised)")
    logger.info("=" * 60)

    model = train_isolation_forest(X_train, contamination=ISO_CONTAMINATION)
    scores = get_anomaly_scores(model, X_train)
    metrics = evaluate_against_labels(scores, y_train.values)
    logger.info(f"Isolation Forest sanity ROC-AUC: {metrics['roc_auc']}, PR-AUC: {metrics['pr_auc']}")
    return model


def phase5(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """Phase 5: XGBoost with validation threshold calibration."""
    logger.info("=" * 60)
    logger.info("Phase 5 — XGBoost (Supervised)")
    logger.info("=" * 60)

    clf = train_xgboost(X_train, y_train, use_smote=True)

    # Calibrate decision threshold on validation set
    optimal_thresh = find_optimal_threshold(clf, X_val, y_val)
    logger.info(f"Calibrated optimal decision threshold: {optimal_thresh:.4f}")

    # Evaluate on unseen test set using both default 0.5 and calibrated threshold
    metrics_default = evaluate_model(clf, X_test, y_test, threshold=0.5)
    metrics_optimal = evaluate_model(clf, X_test, y_test, threshold=optimal_thresh)

    logger.info(f"Test Metrics (Default 0.50 Threshold):")
    logger.info(f"  Precision: {metrics_default['precision']}, Recall: {metrics_default['recall']}, F1: {metrics_default['f1']}")
    logger.info(f"Test Metrics (Calibrated {optimal_thresh:.4f} Threshold):")
    logger.info(f"  Precision: {metrics_optimal['precision']}, Recall: {metrics_optimal['recall']}, F1: {metrics_optimal['f1']}")
    logger.info(f"  ROC-AUC:   {metrics_optimal['roc_auc']}, PR-AUC: {metrics_optimal['pr_auc']}")

    imp = feature_importance(clf, list(X_train.columns))

    # Save metrics report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_content = f"""# Model Evaluation Metrics

## XGBoost (Held-Out Test Set)

### Performance Comparison

| Metric | Default (0.50 Threshold) | Calibrated ({optimal_thresh:.4f} Threshold) |
|--------|--------------------------|---------------------------------------------|
| **Precision** | {metrics_default['precision']:.4f} | **{metrics_optimal['precision']:.4f}** |
| **Recall** | {metrics_default['recall']:.4f} | **{metrics_optimal['recall']:.4f}** |
| **F1-Score** | {metrics_default['f1']:.4f} | **{metrics_optimal['f1']:.4f}** |
| **ROC-AUC** | {metrics_default['roc_auc']:.4f} | **{metrics_optimal['roc_auc']:.4f}** |
| **PR-AUC** | {metrics_default['pr_auc']:.4f} | **{metrics_optimal['pr_auc']:.4f}** |

### Confusion Matrix (Calibrated Threshold)
```
{metrics_optimal['confusion_matrix']}
```

### Feature Importance (Top Features)
| Feature | Importance |
|---------|------------|
"""
    for _, row in imp.head(8).iterrows():
        report_content += f"| `{row['feature']}` | {row['importance']:.4f} |\n"

    (REPORTS_DIR / "model_metrics.md").write_text(report_content, encoding="utf-8")
    logger.info("Saved model metrics -> reports/model_metrics.md")

    # Save pipeline metadata
    metadata = {
        "optimal_threshold": optimal_thresh,
        "test_metrics_calibrated": metrics_optimal,
        "test_metrics_default": metrics_default,
        "top_features": imp.to_dict(orient="records"),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return clf, optimal_thresh


def phase6(X_test: pd.DataFrame, iso_model, xgb_model):
    """Phase 6: Multi-signal Ensemble Scoring."""
    logger.info("=" * 60)
    logger.info("Phase 6 — Ensemble Risk Scoring")
    logger.info("=" * 60)

    result = compute_ensemble(X_test, iso_model=iso_model, xgb_model=xgb_model)
    tier_dist = result["risk_tier"].value_counts()
    for tier in ["Low", "Medium", "High"]:
        logger.info(f"  {tier} Risk: {tier_dist.get(tier, 0):,}")
    return result


def phase7(scored_df: pd.DataFrame):
    """Phase 7: Compliance Mapping."""
    logger.info("=" * 60)
    logger.info("Phase 7 — Compliance Engine")
    logger.info("=" * 60)

    high_risk_samples = scored_df[scored_df["risk_tier"] == "High"].head(5)
    for i, (_, row) in enumerate(high_risk_samples.iterrows(), start=1):
        flags = get_compliance_flags(row["risk_tier"], row.to_dict())
        logger.info(f"Sample High-Risk #{i} (£{row.get('amt', 0):.2f}): {format_flags_text(flags)}")


def main():
    start = time.time()
    logger.info(">>> STARTING FORENSIC FRAUD DETECTION PIPELINE <<<")

    train_df, test_df = phase1()
    phase2()
    X_train, y_train, X_val, y_val, X_test, y_test = phase3(train_df, test_df)
    iso_model = phase4(X_train, y_train)
    xgb_model, threshold = phase5(X_train, y_train, X_val, y_val, X_test, y_test)
    scored_test = phase6(X_test, iso_model, xgb_model)
    phase7(scored_test)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"[SUCCESS] Full Pipeline Completed in {elapsed:.1f}s")
    logger.info("Launch dashboard: streamlit run app/streamlit_app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

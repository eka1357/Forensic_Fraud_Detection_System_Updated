"""run_pipeline.py -- Run the full forensic fraud detection pipeline.

Usage:
    python src/pipeline/run_pipeline.py

This script executes all phases sequentially:
  Phase 1: Data loading & cleaning
  Phase 2: Benford's Law scoring (tested via unit tests)
  Phase 3: Feature engineering
  Phase 4: Isolation Forest training
  Phase 5: XGBoost training & evaluation
  Phase 6: Ensemble risk scoring
  Phase 7: Compliance flag mapping
"""

import sys, os, time
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np


def phase1():
    """Data loading & cleaning."""
    print("=" * 60)
    print("Phase 1 -- Data Understanding & Cleaning")
    print("=" * 60)
    from src.utils.data_loader import prepare_data, load_processed_data
    from src.utils.eda import generate_eda_report

    train_path, test_path = prepare_data()
    print(f"  [OK] Processed train -> {train_path}")
    print(f"  [OK] Processed test  -> {test_path}")

    train_df, test_df = load_processed_data()
    generate_eda_report(train_df, test_df)
    print("  [OK] EDA report generated")

    fraud_rate = train_df["is_fraud"].mean()
    print(f"  [OK] Training fraud rate: {fraud_rate:.4%}")
    print()
    return train_df, test_df


def phase2():
    """Benford's Law module (tested separately via pytest)."""
    print("=" * 60)
    print("Phase 2 -- Benford's Law Module")
    print("=" * 60)
    from src.features.benford import benford_chi2
    # Quick sanity check
    sample = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9] * 100)
    score = benford_chi2(sample)
    print(f"  [OK] Benford module loaded. Sanity chi2 on uniform digits: {score:.1f}")
    print("    (Run `pytest tests/test_benford.py -v` for full unit tests)")
    print()


def phase3(train_df, test_df):
    """Feature engineering."""
    print("=" * 60)
    print("Phase 3 -- Feature Engineering")
    print("=" * 60)
    from src.features.feature_engineering import engineer_features, get_feature_matrix

    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)
    X_train, y_train = get_feature_matrix(train_fe)
    X_test, y_test = get_feature_matrix(test_fe)

    print(f"  [OK] Training features: {X_train.shape}")
    print(f"  [OK] Test features:     {X_test.shape}")
    print(f"  [OK] Feature columns:   {list(X_train.columns)}")
    print("  [OK] Imbalance strategy: SMOTE + class weights (documented in feature_engineering.py)")
    print()
    return X_train, y_train, X_test, y_test


def phase4(X_train):
    """Isolation Forest."""
    print("=" * 60)
    print("Phase 4 -- Isolation Forest (Unsupervised)")
    print("=" * 60)
    from src.models.isolation_forest_model import (
        train_isolation_forest, get_anomaly_scores, evaluate_against_labels,
    )

    model = train_isolation_forest(X_train, contamination=0.006)
    scores = get_anomaly_scores(model, X_train)
    print(f"  [OK] Anomaly scores: min={scores.min():.3f}, max={scores.max():.3f}")
    print("  [OK] Contamination set to 0.006 (~0.6%) to match observed fraud rate.")
    print()
    return model


def phase5(X_train, y_train, X_test, y_test):
    """XGBoost."""
    print("=" * 60)
    print("Phase 5 -- XGBoost (Supervised)")
    print("=" * 60)
    from src.models.xgboost_model import train_xgboost, evaluate_model, feature_importance

    clf = train_xgboost(X_train, y_train, use_smote=True)
    metrics = evaluate_model(clf, X_test, y_test)
    print("  [OK] Test metrics:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"      {k}: {v}")
    print(f"      confusion_matrix: {metrics['confusion_matrix']}")

    imp = feature_importance(clf, list(X_train.columns))
    print("  [OK] Top 5 features:")
    for _, row in imp.head().iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")
    print()

    # Save metrics to reports
    from pathlib import Path
    reports_dir = Path(PROJECT_ROOT) / "reports"
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / "model_metrics.md", "w", encoding="utf-8") as f:
        f.write("# Model Evaluation Metrics\n\n")
        f.write("## XGBoost (Test Set)\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                f.write(f"| {k} | {v} |\n")
        f.write(f"\n### Confusion Matrix\n```\n{metrics['confusion_matrix']}\n```\n")
        f.write("\n### Feature Importance (Top 5)\n")
        f.write("| Feature | Importance |\n|---------|------------|\n")
        for _, row in imp.head().iterrows():
            f.write(f"| {row['feature']} | {row['importance']:.4f} |\n")
    print("  [OK] Metrics saved -> reports/model_metrics.md")
    print()
    return clf


def phase6(X_test):
    """Ensemble scoring."""
    print("=" * 60)
    print("Phase 6 -- Ensemble Risk Scoring")
    print("=" * 60)
    from src.models.ensemble import compute_ensemble

    result = compute_ensemble(X_test)
    tier_dist = result["risk_tier"].value_counts()
    print("  [OK] Risk tier distribution (test set):")
    for tier in ["Low", "Medium", "High"]:
        count = tier_dist.get(tier, 0)
        print(f"      {tier}: {count:,}")
    print()
    return result


def phase7(result):
    """Compliance mapping."""
    print("=" * 60)
    print("Phase 7 -- Compliance Engine")
    print("=" * 60)
    from src.compliance.compliance_engine import get_compliance_flags, format_flags_text

    high_risk = result[result["risk_tier"] == "High"].head(5)
    print("  [OK] Sample compliance flags for top 5 high-risk transactions:")
    for i, (_, row) in enumerate(high_risk.iterrows()):
        flags = get_compliance_flags(row["risk_tier"], row.to_dict())
        print(f"    Transaction {i+1}: {format_flags_text(flags)}")
    print()


def main():
    start = time.time()
    print("\n>>> FORENSIC FRAUD DETECTION PIPELINE <<<\n")

    train_df, test_df = phase1()
    phase2()
    X_train, y_train, X_test, y_test = phase3(train_df, test_df)
    phase4(X_train)
    phase5(X_train, y_train, X_test, y_test)
    result = phase6(X_test)
    phase7(result)

    elapsed = time.time() - start
    print("=" * 60)
    print(f"[DONE] Pipeline complete in {elapsed:.1f}s")
    print("   Run `streamlit run app/streamlit_app.py` to launch the dashboard.")
    print("=" * 60)


if __name__ == "__main__":
    main()

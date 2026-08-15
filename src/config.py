"""config.py — Centralized configuration and path management for the Forensic Fraud Detection System."""

from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT / "Datasets"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Model artifact paths
ISO_MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_fraud.pkl"
METADATA_PATH = MODELS_DIR / "pipeline_metadata.json"

# Dataset column configurations
KEEP_COLS = [
    "trans_date_trans_time", "cc_num", "category", "amt",
    "gender", "city", "state", "zip", "lat", "long",
    "city_pop", "unix_time", "merch_lat", "merch_long", "is_fraud",
]

FEATURE_COLS = [
    "amt", "amt_log", "hour", "dayofweek",
    "lat", "long", "merch_lat", "merch_long",
    "city_pop", "distance", "category_code", "benford_score",
]

LABEL_COL = "is_fraud"

# Modeling Hyperparameters & Weights
ISO_CONTAMINATION = 0.006  # Aligned with empirical ~0.58% training fraud rate
W_XGB = 0.50
W_ISO = 0.30
W_BEN = 0.20

# Risk Tier Percentiles
TIER_P90 = 0.90
TIER_P97 = 0.97

# Compliance Rules Parameters
COMPLIANCE_HIGH_AMOUNT_THRESHOLD = 5000.0
COMPLIANCE_UNUSUAL_HOUR_START = 0
COMPLIANCE_UNUSUAL_HOUR_END = 5

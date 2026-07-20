"""clean_data.py — Standalone data cleaning script (convenience wrapper).

Usage:
    python src/utils/clean_data.py

Loads raw CSVs, cleans them, and saves to data/processed/.
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import prepare_data

if __name__ == "__main__":
    train_path, test_path = prepare_data()
    print(f"Cleaned training data → {train_path}")
    print(f"Cleaned test data     → {test_path}")

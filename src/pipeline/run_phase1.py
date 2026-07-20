"""run_phase1.py — Data Understanding & Cleaning

Loads raw CSVs, cleans them, saves processed versions, and generates the
EDA report with figures.
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import prepare_data, load_processed_data
from src.utils.eda import generate_eda_report


def run():
    print("=" * 60)
    print("Phase 1 — Data Understanding & Cleaning")
    print("=" * 60)

    # Step 1: clean and save
    train_path, test_path = prepare_data()
    print(f"  [OK] Processed training data -> {train_path}")
    print(f"  [OK] Processed test data     -> {test_path}")

    # Step 2: generate EDA report
    train_df, test_df = load_processed_data()
    generate_eda_report(train_df, test_df)

    print("  [OK] EDA report generated")
    print("Phase 1 complete.\n")


if __name__ == "__main__":
    run()

"""clean_data.py — Convenience CLI entry point for data preparation."""

import logging
from src.utils.data_loader import prepare_data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_path, test_path = prepare_data()
    print(f"Cleaned training data -> {train_path}")
    print(f"Cleaned test data     -> {test_path}")

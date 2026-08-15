"""run_phase1.py — Standalone runner for Phase 1: Data Understanding & Cleaning."""

import logging
from src.utils.data_loader import prepare_data, load_processed_data
from src.utils.eda import generate_eda_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase1")


def run():
    logger.info("=" * 60)
    logger.info("Phase 1 — Data Understanding & Cleaning")
    logger.info("=" * 60)

    train_path, test_path = prepare_data()
    logger.info(f"Processed training data -> {train_path}")
    logger.info(f"Processed test data     -> {test_path}")

    train_df, test_df = load_processed_data()
    generate_eda_report(train_df, test_df)

    logger.info("Phase 1 complete.")


if __name__ == "__main__":
    run()

"""
run_eda.py — Script to run full EDA and save results.

Usage:
    python run_eda.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.data_loader import (
    build_dataset_summary,
    load_csv,
    save_dataset_summary,
    validate_schema,
)
from src.eda import run_full_eda

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

TRAIN_CSV = Path("dataset/train.csv")
TEST_CSV  = Path("dataset/test.csv")
REPORTS_DIR = Path("reports")


def main():
    logger.info("=== EDA Pipeline ===")

    # 1. Load data
    df_train = load_csv(TRAIN_CSV)
    df_test  = load_csv(TEST_CSV)

    # 2. Validate schema
    validate_schema(df_train, "train")
    validate_schema(df_test, "test")

    # 3. Build and save dataset summary
    summary = build_dataset_summary(df_train, df_test)
    save_dataset_summary(summary, REPORTS_DIR / "dataset_summary.json")

    # 4. Print quick overview
    print("\n=== TRAIN DATASET ===")
    print(f"Shape: {df_train.shape}")
    print(f"Missing values:\n{df_train.isnull().sum()[df_train.isnull().sum() > 0]}")
    print(f"Duplicates: {df_train.duplicated().sum()}")
    print(f"\nTarget distribution:\n{df_train['satisfaction'].value_counts()}")
    print(f"\nNumerical stats:\n{df_train.describe()}")

    print("\n=== TEST DATASET ===")
    print(f"Shape: {df_test.shape}")
    print(f"Missing values:\n{df_test.isnull().sum()[df_test.isnull().sum() > 0]}")
    print(f"Target distribution:\n{df_test['satisfaction'].value_counts()}")

    # 5. Run all EDA plots
    logger.info("Generating EDA figures …")
    run_full_eda(df_train)

    logger.info("EDA complete. Figures saved to reports/figures/")


if __name__ == "__main__":
    main()

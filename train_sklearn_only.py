"""
train_sklearn_only.py
=====================
Trains Logistic Regression, Random Forest, and HGB models without TensorFlow.
Run this if TF is not yet installed or for faster iteration.

Usage:
    python train_sklearn_only.py
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data_loader import FEATURE_COLS, get_feature_matrix, get_target, load_csv
from src.preprocessing import fit_pipeline, transform
from src.train import (
    MODELS_DIR,
    SEED,
    save_model,
    split_train_val,
    train_hgb,
    train_logistic_regression,
    train_random_forest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

random.seed(SEED)
np.random.seed(SEED)


def main():
    logger.info("=== Sklearn-Only Training Pipeline ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_train = load_csv(Path("dataset/train.csv"))
    X_raw = get_feature_matrix(df_train)
    y = get_target(df_train)

    # Split
    X_tr_raw, X_val_raw, y_tr, y_val = split_train_val(X_raw, y)

    # Fit preprocessing pipeline on training subset only
    pipeline = fit_pipeline(X_tr_raw[FEATURE_COLS])
    X_tr = transform(pipeline, X_tr_raw[FEATURE_COLS])
    X_val_t = transform(pipeline, X_val_raw[FEATURE_COLS])

    # Train models
    lr  = train_logistic_regression(X_tr, y_tr.values)
    rf  = train_random_forest(X_tr, y_tr.values)
    hgb = train_hgb(X_tr, y_tr.values)

    save_model(lr,  "logistic_regression")
    save_model(rf,  "random_forest")
    save_model(hgb, "hgb")

    # Quick validation metrics
    for name, clf in [("LR", lr), ("RF", rf), ("HGB", hgb)]:
        y_prob = clf.predict_proba(X_val_t)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        logger.info(
            "%s — Val AUC=%.4f  F1=%.4f  Acc=%.4f",
            name,
            roc_auc_score(y_val, y_prob),
            f1_score(y_val, y_pred),
            accuracy_score(y_val, y_pred),
        )

    logger.info("Sklearn models trained and saved to models/")


if __name__ == "__main__":
    main()

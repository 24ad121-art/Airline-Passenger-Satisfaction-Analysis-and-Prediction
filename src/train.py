"""
train.py
========
Trains Logistic Regression, Random Forest, HistGradientBoosting, and
a TensorFlow/Keras MLP model on the airline satisfaction dataset.

Usage
-----
    python -m src.train

All models are saved to `models/`.
Metrics are written to `reports/model_metrics.json` and
`reports/model_comparison.csv`.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from src.data_loader import (
    FEATURE_COLS,
    get_feature_matrix,
    get_target,
    load_csv,
)
from src.preprocessing import fit_pipeline, transform

logger = logging.getLogger(__name__)

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
TRAIN_CSV = Path("dataset/train.csv")

# Validation split — 20 % of training data
VAL_FRAC = 0.20


def set_tf_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def split_train_val(
    X: pd.DataFrame, y: pd.Series, val_frac: float = VAL_FRAC
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=SEED)
    train_idx, val_idx = next(sss.split(X, y))
    return X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]


# ── Traditional ML ────────────────────────────────────────────────────────────

def train_logistic_regression(X_tr: np.ndarray, y_tr: np.ndarray) -> LogisticRegression:
    logger.info("Training Logistic Regression …")
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_tr, y_tr)
    return clf


def train_random_forest(X_tr: np.ndarray, y_tr: np.ndarray) -> RandomForestClassifier:
    logger.info("Training Random Forest …")
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    return clf


def train_hgb(X_tr: np.ndarray, y_tr: np.ndarray) -> HistGradientBoostingClassifier:
    logger.info("Training HistGradientBoosting …")
    clf = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=20, random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    return clf


# ── Deep learning (MLP) ───────────────────────────────────────────────────────

def build_mlp(input_dim: int) -> tf.keras.Model:
    import tensorflow as tf
    set_tf_seeds()

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> tuple[tf.keras.Model, dict]:
    import tensorflow as tf
    set_tf_seeds()

    logger.info("Training MLP (TensorFlow/Keras) …")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(MODELS_DIR / "mlp_best.keras")

    model = build_mlp(X_tr.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=10, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_auc", save_best_only=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=512,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info("MLP training complete — best model at %s", checkpoint_path)
    return model, history.history


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model: object, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved %s → %s", name, path)
    return path


def load_model(name: str) -> object:
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_keras_model(path: str | Path | None = None) -> tf.keras.Model:
    import tensorflow as tf
    p = Path(path) if path else MODELS_DIR / "mlp_best.keras"
    return tf.keras.models.load_model(str(p))


# ── Main training pipeline ────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    # 1. Load training data
    df_train = load_csv(TRAIN_CSV)
    X_raw = get_feature_matrix(df_train)
    y = get_target(df_train)

    # 2. Train/val split (stratified)
    X_tr_raw, X_val_raw, y_tr, y_val = split_train_val(X_raw, y)

    # 3. Fit preprocessing pipeline on training subset only
    pipeline = fit_pipeline(X_tr_raw[FEATURE_COLS])
    X_tr = transform(pipeline, X_tr_raw[FEATURE_COLS])
    X_val = transform(pipeline, X_val_raw[FEATURE_COLS])

    # 4. Train traditional models
    lr  = train_logistic_regression(X_tr, y_tr.values)
    rf  = train_random_forest(X_tr, y_tr.values)
    hgb = train_hgb(X_tr, y_tr.values)

    save_model(lr, "logistic_regression")
    save_model(rf, "random_forest")
    save_model(hgb, "hgb")

    # 5. Train MLP
    mlp, history = train_mlp(X_tr, y_tr.values, X_val, y_val.values)
    # Save MLP history for dashboard
    hist_path = MODELS_DIR / "mlp_history.json"
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)

    # 6. Save validation predictions for quick eval
    val_preds = {
        "logistic_regression": lr.predict_proba(X_val)[:, 1],
        "random_forest":       rf.predict_proba(X_val)[:, 1],
        "hgb":                 hgb.predict_proba(X_val)[:, 1],
        "mlp":                 mlp.predict(X_val, verbose=0).flatten(),
        "y_val":               y_val.values,
    }
    np.save(MODELS_DIR / "val_predictions.npy", val_preds)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()

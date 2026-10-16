"""
preprocessing.py
================
Reusable sklearn Pipeline for the airline satisfaction dataset.

Design decisions
----------------
* ID columns (Unnamed: 0, id) are excluded from features.
* Target column is never used as a feature.
* Median imputation for Arrival Delay in Minutes (fitted on train only).
* OrdinalEncoder with handle_unknown='use_encoded_value' / unknown_value=-1
  so unseen categories produce a deterministic sentinel rather than crashing.
* StandardScaler applied to numerical features for models that need it.
* The same pipeline is used during training and inference to guarantee a
  consistent feature schema.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.data_loader import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERICAL_COLS,
    RATING_COLS,
)

logger = logging.getLogger(__name__)

PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")


# ── Build pipeline ───────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """Construct (but do not fit) the full preprocessing Pipeline."""

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Rating columns (0-5 integers) — impute zeros are valid, no scaling needed
    rating_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_COLS),
            ("num", numerical_transformer, NUMERICAL_COLS),
            ("rat", rating_transformer, RATING_COLS),
        ],
        remainder="drop",  # silently drops ID cols / target if still present
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    return pipeline


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Return ordered feature names after transformation."""
    ct: ColumnTransformer = pipeline.named_steps["preprocessor"]
    names: list[str] = []
    for name, transformer, cols in ct.transformers_:
        if name == "remainder":
            continue
        if isinstance(cols, list):
            names.extend(cols)
        else:
            names.extend(list(cols))
    return names


def fit_pipeline(X_train: pd.DataFrame, save: bool = True) -> Pipeline:
    """Fit the preprocessing pipeline on the training features and optionally save it."""
    # Keep only known feature columns to avoid ID/target leakage
    X_train = X_train[FEATURE_COLS]
    pipeline = build_pipeline()
    pipeline.fit(X_train)
    logger.info("Pipeline fitted on %d rows × %d features", *X_train.shape)
    if save:
        save_pipeline(pipeline)
    return pipeline


def transform(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply the fitted pipeline to a feature DataFrame."""
    X = X[FEATURE_COLS]
    return pipeline.transform(X)


def fit_transform(X_train: pd.DataFrame, save: bool = True) -> tuple[Pipeline, np.ndarray]:
    """Convenience: fit and immediately transform the training data."""
    pipeline = fit_pipeline(X_train, save=save)
    X_transformed = transform(pipeline, X_train)
    return pipeline, X_transformed


# ── Persistence ───────────────────────────────────────────────────────────────

def save_pipeline(pipeline: Pipeline, path: str | Path = PIPELINE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Preprocessing pipeline saved → %s", path)


def load_pipeline(path: str | Path = PIPELINE_PATH) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline not found: {path}. Run training first.")
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    logger.info("Preprocessing pipeline loaded ← %s", path)
    return pipeline


# ── Single-record inference helper ────────────────────────────────────────────

def record_to_dataframe(record: dict) -> pd.DataFrame:
    """
    Convert a dict of passenger attributes (as supplied by the Streamlit form
    or unit tests) into a one-row DataFrame with the correct feature schema.

    Missing keys receive NaN so the pipeline imputers handle them gracefully.
    """
    row = {col: record.get(col, np.nan) for col in FEATURE_COLS}
    return pd.DataFrame([row])

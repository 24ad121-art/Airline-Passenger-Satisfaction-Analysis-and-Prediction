"""
predict.py
==========
Single-record and batch inference using saved preprocessing pipeline
and model artifacts. The dashboard uses this module for live predictions.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_loader import FEATURE_COLS, POSITIVE_CLASS
from src.preprocessing import load_pipeline, record_to_dataframe, transform
from src.train import load_keras_model, load_model

logger = logging.getLogger(__name__)

# Default final model (can be overridden)
DEFAULT_MODEL = "hgb"
THRESHOLD = 0.50


# ── Public API ────────────────────────────────────────────────────────────────

def predict_single(record: dict, model_name: str = DEFAULT_MODEL) -> dict:
    """
    Predict satisfaction for a single passenger record.

    Parameters
    ----------
    record : dict
        Keys should match FEATURE_COLS. Missing keys are imputed by the pipeline.
    model_name : str
        One of 'logistic_regression', 'random_forest', 'hgb', 'mlp'.

    Returns
    -------
    dict with keys:
        predicted_class  : str   — 'satisfied' or 'neutral or dissatisfied'
        probability_satisfied : float
        probability_dissatisfied : float
        model_used : str
    """
    pipeline = load_pipeline()
    df_single = record_to_dataframe(record)
    X = transform(pipeline, df_single)

    prob_satisfied = _get_prob(model_name, X)
    predicted_class = POSITIVE_CLASS if prob_satisfied >= THRESHOLD else "neutral or dissatisfied"

    return {
        "predicted_class":          predicted_class,
        "probability_satisfied":    round(float(prob_satisfied), 4),
        "probability_dissatisfied": round(float(1 - prob_satisfied), 4),
        "model_used":               model_name,
        "threshold":                THRESHOLD,
    }


def predict_batch(df: pd.DataFrame, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Return predicted probability of satisfaction for each row in df."""
    pipeline = load_pipeline()
    X = transform(pipeline, df[FEATURE_COLS])
    return _get_prob_batch(model_name, X)


def _get_prob(model_name: str, X: np.ndarray) -> float:
    return float(_get_prob_batch(model_name, X)[0])


def _get_prob_batch(model_name: str, X: np.ndarray) -> np.ndarray:
    if model_name == "mlp":
        model = load_keras_model()
        return model.predict(X, verbose=0).flatten()
    clf = load_model(model_name)
    return clf.predict_proba(X)[:, 1]


# ── CLI helper ────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    example = {
        "Gender":                            "Male",
        "Customer Type":                     "Loyal Customer",
        "Age":                               35,
        "Type of Travel":                    "Business travel",
        "Class":                             "Business",
        "Flight Distance":                   1200,
        "Inflight wifi service":             4,
        "Departure/Arrival time convenient": 3,
        "Ease of Online booking":            4,
        "Gate location":                     3,
        "Food and drink":                    4,
        "Online boarding":                   4,
        "Seat comfort":                      5,
        "Inflight entertainment":            5,
        "On-board service":                  4,
        "Leg room service":                  4,
        "Baggage handling":                  4,
        "Checkin service":                   4,
        "Inflight service":                  5,
        "Cleanliness":                       5,
        "Departure Delay in Minutes":        0,
        "Arrival Delay in Minutes":          0,
    }
    result = predict_single(example)
    print("\n=== Prediction Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

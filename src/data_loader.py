"""
data_loader.py
==============
Handles loading, validating, and summarising the airline satisfaction datasets.
All operations are read-only — original files are never modified.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Column schema ────────────────────────────────────────────────────────────
ID_COLS: list[str] = ["Unnamed: 0", "id"]
TARGET_COL: str = "satisfaction"
POSITIVE_CLASS: str = "satisfied"
NEGATIVE_CLASS: str = "neutral or dissatisfied"

CATEGORICAL_COLS: list[str] = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

RATING_COLS: list[str] = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]

NUMERICAL_COLS: list[str] = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

FEATURE_COLS: list[str] = CATEGORICAL_COLS + RATING_COLS + NUMERICAL_COLS

EXPECTED_COLUMNS: set[str] = set(ID_COLS + FEATURE_COLS + [TARGET_COL])


# ── Public helpers ────────────────────────────────────────────────────────────

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame without altering any values."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %s  shape=%s", path.name, df.shape)
    return df


def validate_schema(df: pd.DataFrame, name: str = "dataset") -> None:
    """Raise ValueError if expected columns are missing."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")
    logger.info("[%s] Schema OK — %d columns present", name, len(df.columns))


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the predictive feature columns only (drops IDs and target)."""
    drop = [c for c in ID_COLS + [TARGET_COL] if c in df.columns]
    return df.drop(columns=drop)


def get_target(df: pd.DataFrame) -> pd.Series:
    """Return the binary target series (1 = satisfied, 0 = neutral/dissatisfied)."""
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found.")
    return (df[TARGET_COL] == POSITIVE_CLASS).astype(int).rename(TARGET_COL)


def build_dataset_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Return a machine-readable summary dict for both splits."""

    def _split_summary(df: pd.DataFrame, name: str) -> dict[str, Any]:
        target_counts = df[TARGET_COL].value_counts().to_dict()
        n = len(df)
        return {
            "name": name,
            "rows": n,
            "columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "target_distribution": {k: {"count": v, "pct": round(v / n * 100, 2)}
                                    for k, v in target_counts.items()},
            "numerical_stats": df[NUMERICAL_COLS].describe().round(4).to_dict(),
            "rating_stats": df[RATING_COLS].describe().round(4).to_dict(),
            "unique_values": {c: int(df[c].nunique()) for c in df.columns},
        }

    summary: dict[str, Any] = {
        "train": _split_summary(train, "train"),
        "test": _split_summary(test, "test"),
        "feature_columns": FEATURE_COLS,
        "id_columns": ID_COLS,
        "target_column": TARGET_COL,
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
    }
    return summary


def save_dataset_summary(summary: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Dataset summary saved → %s", output_path)

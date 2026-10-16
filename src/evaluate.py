"""
evaluate.py
===========
Evaluates trained models on the held-out test set and saves metrics.

Usage
-----
    python -m src.evaluate

Reads:  dataset/test.csv
Writes: reports/model_metrics.json
        reports/model_comparison.csv
        reports/figures/eval_*.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data_loader import get_feature_matrix, get_target, load_csv
from src.preprocessing import load_pipeline, transform
from src.train import MODELS_DIR, REPORTS_DIR, load_keras_model, load_model

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

FIGURES_DIR = REPORTS_DIR / "figures"
TEST_CSV = Path("dataset/test.csv")
POSITIVE_CLASS_LABEL = "satisfied"


# ── Core metric computation ───────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":       round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":          round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":         round(float(roc_auc_score(y_true, y_prob)), 4),
        "avg_precision":   round(float(average_precision_score(y_true, y_prob)), 4),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["neutral/dissatisfied", "satisfied"]
        ),
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

PALETTE = {
    "Logistic Regression": "#3498db",
    "Random Forest":       "#2ecc71",
    "HGB":                 "#e67e22",
    "MLP":                 "#9b59b6",
}


def plot_roc_curves(
    results: dict[str, dict], y_true: np.ndarray, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_true, r["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['metrics']['roc_auc']:.4f})",
                color=PALETTE.get(name, None))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = output_dir / "eval_roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved → %s", path)


def plot_pr_curves(
    results: dict[str, dict], y_true: np.ndarray, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in results.items():
        precision, recall, _ = precision_recall_curve(y_true, r["y_prob"])
        ap = r["metrics"]["avg_precision"]
        ax.plot(recall, precision, label=f"{name} (AP={ap:.4f})",
                color=PALETTE.get(name, None))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Test Set")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = output_dir / "eval_pr_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curves saved → %s", path)


def plot_confusion_matrices(
    results: dict[str, dict], y_true: np.ndarray, output_dir: Path
) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        y_pred = (r["y_prob"] >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm,
                                      display_labels=["neutral/dissat.", "satisfied"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10)
    fig.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = output_dir / "eval_confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrices saved → %s", path)


def plot_model_comparison_bar(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(metrics))
    width = 0.18
    models = comparison_df["model"].tolist()
    for i, row in comparison_df.iterrows():
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["model"],
               color=PALETTE.get(row["model"], None), edgecolor="white")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.legend(title="Model")
    fig.tight_layout()
    path = output_dir / "eval_model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison bar chart saved → %s", path)


def plot_mlp_history(output_dir: Path) -> None:
    hist_path = MODELS_DIR / "mlp_history.json"
    if not hist_path.exists():
        logger.warning("MLP history not found — skipping history plot")
        return
    with open(hist_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("MLP — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="Train Acc")
    if "val_accuracy" in history:
        axes[1].plot(history["val_accuracy"], label="Val Acc")
    axes[1].set_title("MLP — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    path = output_dir / "eval_mlp_history.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("MLP history plot saved → %s", path)


# ── Main evaluation ───────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load test data
    df_test = load_csv(TEST_CSV)
    X_test_raw = get_feature_matrix(df_test)
    y_test = get_target(df_test).values

    # 2. Load and apply preprocessing
    pipeline = load_pipeline()
    X_test = transform(pipeline, X_test_raw)

    # 3. Load models and compute predictions
    results: dict[str, dict] = {}

    for model_key, display_name in [
        ("logistic_regression", "Logistic Regression"),
        ("random_forest",       "Random Forest"),
        ("hgb",                 "HGB"),
    ]:
        clf = load_model(model_key)
        y_prob = clf.predict_proba(X_test)[:, 1]
        results[display_name] = {
            "y_prob":  y_prob,
            "metrics": compute_metrics(y_test, y_prob),
        }
        logger.info("%s — AUC=%.4f  F1=%.4f",
                    display_name,
                    results[display_name]["metrics"]["roc_auc"],
                    results[display_name]["metrics"]["f1_score"])

    # MLP
    try:
        mlp = load_keras_model()
        y_prob_mlp = mlp.predict(X_test, verbose=0).flatten()
        results["MLP"] = {
            "y_prob":  y_prob_mlp,
            "metrics": compute_metrics(y_test, y_prob_mlp),
        }
        logger.info("MLP — AUC=%.4f  F1=%.4f",
                    results["MLP"]["metrics"]["roc_auc"],
                    results["MLP"]["metrics"]["f1_score"])
    except (OSError, ImportError, ValueError, RuntimeError, AttributeError) as e:
        logger.warning("Could not load MLP: %s", e)

    # 4. Plots
    plot_roc_curves(results, y_test, FIGURES_DIR)
    plot_pr_curves(results, y_test, FIGURES_DIR)
    plot_confusion_matrices(results, y_test, FIGURES_DIR)
    plot_mlp_history(FIGURES_DIR)

    # 5. Comparison table
    rows = []
    for name, r in results.items():
        m = r["metrics"]
        rows.append({
            "model":         name,
            "accuracy":      m["accuracy"],
            "precision":     m["precision"],
            "recall":        m["recall"],
            "f1_score":      m["f1_score"],
            "roc_auc":       m["roc_auc"],
            "avg_precision": m["avg_precision"],
        })
    comparison_df = pd.DataFrame(rows)
    plot_model_comparison_bar(comparison_df, FIGURES_DIR)

    # 6. Save metrics
    metrics_out: dict = {}
    for name, r in results.items():
        metrics_out[name] = {k: v for k, v in r["metrics"].items() if k != "classification_report"}
        metrics_out[name]["classification_report"] = r["metrics"]["classification_report"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    logger.info("Evaluation complete. Metrics saved to reports/")


if __name__ == "__main__":
    main()

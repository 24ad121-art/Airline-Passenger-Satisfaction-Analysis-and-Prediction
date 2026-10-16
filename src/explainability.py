"""
explainability.py
=================
Global and local feature-importance analysis using SHAP and sklearn
built-in feature importances.

Usage
-----
    python -m src.explainability

Outputs saved to reports/figures/ and reports/model_metrics.json (updated).

Limitations
-----------
* SHAP values indicate the model's reliance on a feature for predictions,
  not a causal relationship between the feature and actual satisfaction.
* TreeExplainer is exact for tree-based models; for the MLP, a background
  sample is used which gives approximate (kernel-based) SHAP values.
* Feature importance can be unstable for correlated features.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import get_feature_matrix, load_csv
from src.preprocessing import get_feature_names, load_pipeline, transform
from src.train import REPORTS_DIR, load_model

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

FIGURES_DIR = REPORTS_DIR / "figures"
TRAIN_CSV = Path("dataset/train.csv")

# Background sample size for SHAP
BG_SAMPLE = 200
SHAP_SAMPLE = 500


# ── Helper ────────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", path)


# ── Sklearn-native importance ─────────────────────────────────────────────────

def plot_sklearn_importance(model, feature_names: list[str], model_name: str) -> None:
    """Plot feature importance from tree-based models (MDI) or LR coefficients."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"{model_name} — Feature Importances (MDI)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = f"{model_name} — |Coefficient| Importance"
    else:
        logger.warning("No importances available for %s", model_name)
        return

    idx = np.argsort(importances)[::-1][:20]
    idx_rev = idx[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([feature_names[i] for i in idx_rev],
            importances[idx_rev], color="#3498db", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    _save_fig(fig, f"expl_{safe_name}_importance.png")


# ── SHAP ─────────────────────────────────────────────────────────────────────

def compute_shap_hgb(
    hgb_model,
    X_sample: np.ndarray,
    feature_names: list[str],
) -> None:
    try:
        import shap
        explainer = shap.TreeExplainer(hgb_model)
        shap_values = explainer.shap_values(X_sample)

        # Summary bar
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          plot_type="bar", show=False, max_display=15)
        plt.title("SHAP Global Feature Importance (HGB)", fontsize=12)
        _save_fig(plt.gcf(), "expl_shap_hgb_bar.png")

        # Beeswarm
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          show=False, max_display=15)
        plt.title("SHAP Beeswarm — HGB", fontsize=12)
        _save_fig(plt.gcf(), "expl_shap_hgb_beeswarm.png")

        # Save top-10 feature importance to JSON
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        shap_importance = {feature_names[i]: float(mean_abs[i]) for i in top_idx}
        out_path = REPORTS_DIR / "shap_importance.json"
        with open(out_path, "w") as f:
            json.dump(shap_importance, f, indent=2)
        logger.info("SHAP importance saved → %s", out_path)

    except ImportError:
        logger.warning("SHAP not installed — skipping SHAP analysis")
    except (RuntimeError, ValueError, KeyError, AttributeError, TypeError) as e:
        logger.error("SHAP error: %s", e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    # Load data
    df = load_csv(TRAIN_CSV)
    X_raw = get_feature_matrix(df)

    # Load pipeline
    pipeline = load_pipeline()
    feature_names = get_feature_names(pipeline)
    X_transformed = transform(pipeline, X_raw)

    # Sample for SHAP efficiency
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_transformed), size=min(SHAP_SAMPLE, len(X_transformed)),
                     replace=False)
    X_sample = X_transformed[idx]

    # Load models
    lr  = load_model("logistic_regression")
    rf  = load_model("random_forest")
    hgb = load_model("hgb")

    # Sklearn-native importance plots
    plot_sklearn_importance(lr,  feature_names, "Logistic Regression")
    plot_sklearn_importance(rf,  feature_names, "Random Forest")
    plot_sklearn_importance(hgb, feature_names, "HGB")

    # SHAP (on HGB as primary model — if HGB wins; otherwise swap)
    compute_shap_hgb(hgb, X_sample, feature_names)

    logger.info("Explainability analysis complete.")


if __name__ == "__main__":
    main()

"""
eda.py
======
Reusable EDA functions for the airline passenger satisfaction dataset.
All functions accept DataFrames and output paths, write chart files to
`reports/figures/`, and return the figure objects for notebook use.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("reports/figures")
PALETTE = {"satisfied": "#2ecc71", "neutral or dissatisfied": "#e74c3c"}
STYLE = "whitegrid"

sns.set_theme(style=STYLE, palette="muted", font_scale=1.1)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str, output_dir: Path = FIGURES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / name
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", fpath)
    return fpath


def _fmt_pct(ax: plt.Axes) -> None:
    """Add percentage labels on bar patches."""
    total = sum(p.get_height() for p in ax.patches)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h / total * 100:.1f}%",
                (p.get_x() + p.get_width() / 2, h),
                ha="center", va="bottom", fontsize=8,
            )


# ── 1. Dataset overview ───────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    counts = df["satisfaction"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].bar(counts.index, counts.values,
                color=[PALETTE[k] for k in counts.index], edgecolor="white", linewidth=0.8)
    for i, (label, val) in enumerate(counts.items()):
        axes[0].text(i, val + 300, f"{val:,}\n({val/len(df)*100:.1f}%)",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Target Distribution (Count)")
    axes[0].set_xlabel("Satisfaction")
    axes[0].set_ylabel("Count")
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=[PALETTE[k] for k in counts.index],
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Target Distribution (Share)")
    fig.suptitle("Passenger Satisfaction Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "01_target_distribution.png", output_dir)
    return fig


# ── 2. Passenger demographics ─────────────────────────────────────────────────

def plot_age_distribution(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["Age"], bins=40, color="#3498db", edgecolor="white", linewidth=0.6)
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")

    df2 = df.copy()
    df2["Age Group"] = pd.cut(df2["Age"], bins=[0, 18, 30, 45, 60, 100],
                               labels=["<18", "18-30", "31-45", "46-60", "60+"])
    sat_rate = (df2.groupby("Age Group", observed=True)["satisfaction"]
                .apply(lambda s: (s == "satisfied").mean() * 100)
                .reset_index(name="Satisfaction Rate (%)"))
    axes[1].bar(sat_rate["Age Group"].astype(str), sat_rate["Satisfaction Rate (%)"],
                color="#2ecc71", edgecolor="white")
    axes[1].set_title("Satisfaction Rate by Age Group")
    axes[1].set_xlabel("Age Group")
    axes[1].set_ylabel("Satisfaction Rate (%)")
    axes[1].set_ylim(0, 100)
    fig.tight_layout()
    _save(fig, "02_age_distribution.png", output_dir)
    return fig


def plot_satisfaction_by_gender(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    ct = pd.crosstab(df["Gender"], df["satisfaction"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    ct.plot(kind="bar", ax=ax, color=[PALETTE[c] for c in ct.columns],
            edgecolor="white", linewidth=0.8, rot=0)
    ax.set_title("Satisfaction Rate by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Satisfaction")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    _save(fig, "03_satisfaction_by_gender.png", output_dir)
    return fig


def plot_satisfaction_by_customer_type(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    ct = pd.crosstab(df["Customer Type"], df["satisfaction"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", ax=ax, color=[PALETTE[c] for c in ct.columns],
            edgecolor="white", linewidth=0.8, rot=0)
    ax.set_title("Satisfaction Rate by Customer Type")
    ax.set_xlabel("Customer Type")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Satisfaction")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    _save(fig, "04_satisfaction_by_customer_type.png", output_dir)
    return fig


# ── 3. Travel characteristics ─────────────────────────────────────────────────

def plot_satisfaction_by_travel_type(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    ct = pd.crosstab(df["Type of Travel"], df["satisfaction"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", ax=ax, color=[PALETTE[c] for c in ct.columns],
            edgecolor="white", linewidth=0.8, rot=0)
    ax.set_title("Satisfaction Rate by Type of Travel")
    ax.set_xlabel("Type of Travel")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Satisfaction")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    _save(fig, "05_satisfaction_by_travel_type.png", output_dir)
    return fig


def plot_satisfaction_by_class(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    ct = pd.crosstab(df["Class"], df["satisfaction"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", ax=ax, color=[PALETTE[c] for c in ct.columns],
            edgecolor="white", linewidth=0.8, rot=0)
    ax.set_title("Satisfaction Rate by Travel Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Satisfaction")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    _save(fig, "06_satisfaction_by_class.png", output_dir)
    return fig


def plot_class_travel_heatmap(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    pivot = df.groupby(["Class", "Type of Travel"])["satisfaction"] \
              .apply(lambda s: (s == "satisfied").mean() * 100) \
              .unstack()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax,
                linewidths=0.5, vmin=0, vmax=100, cbar_kws={"label": "Satisfaction Rate (%)"})
    ax.set_title("Satisfaction Rate (%) by Class and Travel Type")
    ax.set_xlabel("Type of Travel")
    ax.set_ylabel("Class")
    fig.tight_layout()
    _save(fig, "07_class_travel_heatmap.png", output_dir)
    return fig


def plot_flight_distance_distribution(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for sat, grp in df.groupby("satisfaction"):
        axes[0].hist(grp["Flight Distance"], bins=50, alpha=0.65,
                     label=sat, color=PALETTE[sat], edgecolor="none")
    axes[0].set_title("Flight Distance by Satisfaction")
    axes[0].set_xlabel("Flight Distance (miles)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    df2 = df.copy()
    df2["Distance Bin"] = pd.cut(df2["Flight Distance"],
                                  bins=[0, 500, 1000, 2000, 5000],
                                  labels=["<500", "500-1000", "1000-2000", ">2000"])
    ct = pd.crosstab(df2["Distance Bin"], df2["satisfaction"], normalize="index") * 100
    ct.plot(kind="bar", ax=axes[1], color=[PALETTE[c] for c in ct.columns],
            edgecolor="white", linewidth=0.8, rot=0)
    axes[1].set_title("Satisfaction Rate by Flight Distance Bin")
    axes[1].set_xlabel("Distance (miles)")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].legend(title="Satisfaction")
    axes[1].set_ylim(0, 100)
    fig.tight_layout()
    _save(fig, "08_flight_distance_distribution.png", output_dir)
    return fig


# ── 4. Service ratings ────────────────────────────────────────────────────────

RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Inflight service", "Cleanliness",
]


def plot_service_ratings_overview(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    means = df.groupby("satisfaction")[RATING_COLS].mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(RATING_COLS))
    width = 0.35
    for i, (sat, row) in enumerate(means.iterrows()):
        ax.bar(x + (i - 0.5) * width, row.values, width,
               label=sat, color=PALETTE[sat], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([c[:18] for c in RATING_COLS], rotation=45, ha="right", fontsize=8)
    ax.set_title("Mean Service Ratings by Satisfaction Group")
    ax.set_ylabel("Mean Rating (0–5)")
    ax.set_ylim(0, 5.5)
    ax.legend(title="Satisfaction")
    fig.tight_layout()
    _save(fig, "09_service_ratings_overview.png", output_dir)
    return fig


def plot_rating_distributions(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    n = len(RATING_COLS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes_flat = axes.flatten()
    for i, col in enumerate(RATING_COLS):
        ax = axes_flat[i]
        ct = pd.crosstab(df[col], df["satisfaction"], normalize="index") * 100
        ct.plot(kind="bar", ax=ax, color=[PALETTE[c] for c in ct.columns],
                edgecolor="white", linewidth=0.5, rot=0, legend=(i == 0))
        ax.set_title(col[:25], fontsize=8)
        ax.set_xlabel("Rating")
        ax.set_ylabel("%" if i % ncols == 0 else "")
        ax.set_ylim(0, 105)
        if i == 0:
            ax.legend(title="Satisfaction", fontsize=7)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Satisfaction Rate by Service Rating Value", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "10_rating_distributions.png", output_dir)
    return fig


# ── 5. Flight delays ──────────────────────────────────────────────────────────

def plot_delay_distributions(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, col in zip(axes[0], ["Departure Delay in Minutes", "Arrival Delay in Minutes"]):
        clipped = df[col].clip(upper=df[col].quantile(0.99))
        ax.hist(clipped, bins=60, color="#3498db", edgecolor="none", alpha=0.8)
        ax.set_title(f"{col} Distribution (≤99th pct)")
        ax.set_xlabel("Minutes")
        ax.set_ylabel("Count")

    for ax, col in zip(axes[1], ["Departure Delay in Minutes", "Arrival Delay in Minutes"]):
        df2 = df.copy()
        df2["Delay Bin"] = pd.cut(df2[col],
                                   bins=[-1, 0, 15, 45, 120, df2[col].max() + 1],
                                   labels=["None", "1-15", "16-45", "46-120", ">120"])
        ct = pd.crosstab(df2["Delay Bin"], df2["satisfaction"], normalize="index") * 100
        ct.plot(kind="bar", ax=ax, color=[PALETTE[c] for c in ct.columns],
                edgecolor="white", linewidth=0.5, rot=0)
        ax.set_title(f"Satisfaction by {col[:20]} Bin")
        ax.set_xlabel("Delay (minutes)")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 100)
        ax.legend(title="Satisfaction", fontsize=7)

    fig.tight_layout()
    _save(fig, "11_delay_distributions.png", output_dir)
    return fig


def plot_delay_scatter(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 6))
    for sat, grp in sample.groupby("satisfaction"):
        ax.scatter(grp["Departure Delay in Minutes"], grp["Arrival Delay in Minutes"],
                   alpha=0.3, s=10, label=sat, color=PALETTE[sat])
    ax.set_xlim(0, sample["Departure Delay in Minutes"].quantile(0.99))
    ax.set_ylim(0, sample["Arrival Delay in Minutes"].quantile(0.99))
    ax.set_title("Departure vs Arrival Delay (sample, ≤99th pct)")
    ax.set_xlabel("Departure Delay (min)")
    ax.set_ylabel("Arrival Delay (min)")
    ax.legend(title="Satisfaction")
    fig.tight_layout()
    _save(fig, "12_delay_scatter.png", output_dir)
    return fig


# ── 6. Statistical / correlation ─────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    num_cols = RATING_COLS + ["Age", "Flight Distance",
                               "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.4, vmin=-1, vmax=1,
                annot_kws={"size": 7})
    ax.set_title("Pearson Correlation Heatmap (Numerical Features)", fontsize=13)
    fig.tight_layout()
    _save(fig, "13_correlation_heatmap.png", output_dir)
    return fig


def plot_satisfaction_rates_summary(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> plt.Figure:
    groups = {
        "Gender": "Gender",
        "Customer Type": "Customer Type",
        "Type of Travel": "Type of Travel",
        "Class": "Class",
    }
    records = []
    for label, col in groups.items():
        for val, grp in df.groupby(col):
            rate = (grp["satisfaction"] == "satisfied").mean() * 100
            records.append({"Group": f"{label}: {val}", "Satisfaction Rate (%)": rate})
    summary_df = pd.DataFrame(records).sort_values("Satisfaction Rate (%)", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(summary_df["Group"], summary_df["Satisfaction Rate (%)"],
            color=["#2ecc71" if v >= 50 else "#e74c3c"
                   for v in summary_df["Satisfaction Rate (%)"]],
            edgecolor="white")
    ax.axvline(50, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Satisfaction Rate by Categorical Group")
    ax.set_xlabel("Satisfaction Rate (%)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    _save(fig, "14_satisfaction_rates_summary.png", output_dir)
    return fig


def run_full_eda(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Run all EDA plots and save to output_dir."""
    output_dir = Path(output_dir)
    logger.info("Running full EDA — output dir: %s", output_dir)
    plot_target_distribution(df, output_dir)
    plot_age_distribution(df, output_dir)
    plot_satisfaction_by_gender(df, output_dir)
    plot_satisfaction_by_customer_type(df, output_dir)
    plot_satisfaction_by_travel_type(df, output_dir)
    plot_satisfaction_by_class(df, output_dir)
    plot_class_travel_heatmap(df, output_dir)
    plot_flight_distance_distribution(df, output_dir)
    plot_service_ratings_overview(df, output_dir)
    plot_rating_distributions(df, output_dir)
    plot_delay_distributions(df, output_dir)
    plot_delay_scatter(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_satisfaction_rates_summary(df, output_dir)
    logger.info("EDA complete — %d figures generated", 14)

"""
app.py — Airline Passenger Satisfaction Dashboard
==================================================
Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Airline Passenger Satisfaction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_CSV   = Path("dataset/train.csv")
REPORTS_DIR = Path("reports")
MODELS_DIR  = Path("models")
FIGURES_DIR = REPORTS_DIR / "figures"

# ── Color palette ─────────────────────────────────────────────────────────────
COLOR_SAT   = "#2ecc71"
COLOR_DISSAT = "#e74c3c"
PALETTE = {"satisfied": COLOR_SAT, "neutral or dissatisfied": COLOR_DISSAT}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #e94560;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #a0aec0;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.section-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e2e8f0;
    border-left: 4px solid #e94560;
    padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
}

.insight-box {
    background: rgba(46, 204, 113, 0.1);
    border-left: 4px solid #2ecc71;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #d1fae5;
}

.pred-satisfied {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 2px solid #2ecc71;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.pred-dissatisfied {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 2px solid #e74c3c;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    background: #1a1a2e;
    color: #a0aec0;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #e94560 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset …")
def load_data() -> pd.DataFrame:
    if not TRAIN_CSV.exists():
        st.error("Training data not found. Place `train.csv` in `dataset/`.")
        st.stop()
    return pd.read_csv(TRAIN_CSV)


@st.cache_resource(show_spinner="Loading models …")
def load_artifacts():
    artifacts: dict = {}
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.pkl"
    if pipeline_path.exists():
        import pickle
        with open(pipeline_path, "rb") as f:
            artifacts["pipeline"] = pickle.load(f)
    for name in ["logistic_regression", "random_forest", "hgb"]:
        p = MODELS_DIR / f"{name}.pkl"
        if p.exists():
            import pickle
            with open(p, "rb") as f:
                artifacts[name] = pickle.load(f)
    mlp_path = MODELS_DIR / "mlp_best.keras"
    if mlp_path.exists():
        try:
            import tensorflow as tf
            artifacts["mlp"] = tf.keras.models.load_model(str(mlp_path))
        except (ImportError, OSError, ValueError, RuntimeError) as e:
            warnings.warn(f"Could not load MLP model: {e}", UserWarning, stacklevel=2)
    return artifacts


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    p = REPORTS_DIR / "model_metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame | None:
    p = REPORTS_DIR / "model_comparison.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def load_mlp_history() -> dict | None:
    p = MODELS_DIR / "mlp_history.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("## ✈️ Navigation")
        st.markdown("---")

        st.markdown("### 🔍 Global Filters")
        gender_opts = ["All"] + sorted(df["Gender"].unique().tolist())
        gender = st.selectbox("Gender", gender_opts)

        class_opts = ["All"] + sorted(df["Class"].unique().tolist())
        travel_class = st.selectbox("Travel Class", class_opts)

        travel_opts = ["All"] + sorted(df["Type of Travel"].unique().tolist())
        travel_type = st.selectbox("Type of Travel", travel_opts)

        age_range = st.slider("Age Range", int(df["Age"].min()),
                               int(df["Age"].max()),
                               (int(df["Age"].min()), int(df["Age"].max())))

        st.markdown("---")
        st.markdown(
            "**Dataset:** Airline Passenger Satisfaction  \n"
            "**Source:** Kaggle  \n"
            f"**Records:** {len(df):,}"
        )

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if gender != "All":
        mask &= df["Gender"] == gender
    if travel_class != "All":
        mask &= df["Class"] == travel_class
    if travel_type != "All":
        mask &= df["Type of Travel"] == travel_type
    mask &= (df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])
    return df[mask]


# ── Helper plot builders ──────────────────────────────────────────────────────

def plotly_bar_satisfaction(df: pd.DataFrame, col: str, title: str,
                             normalize: bool = True) -> go.Figure:
    ct = pd.crosstab(df[col], df["satisfaction"])
    if normalize:
        ct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for sat in ct.columns:
        color = COLOR_SAT if sat == "satisfied" else COLOR_DISSAT
        fig.add_trace(go.Bar(name=sat, x=ct.index.astype(str),
                             y=ct[sat], marker_color=color))
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title=col,
        yaxis_title="%" if normalize else "Count",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#e2e8f0",
        legend_title="Satisfaction",
    )
    return fig


def plotly_histogram(df: pd.DataFrame, col: str, title: str,
                      bins: int = 50) -> go.Figure:
    fig = go.Figure()
    for sat, grp in df.groupby("satisfaction"):
        color = COLOR_SAT if sat == "satisfied" else COLOR_DISSAT
        fig.add_trace(go.Histogram(x=grp[col], name=sat, opacity=0.7,
                                   marker_color=color, nbinsx=bins))
    fig.update_layout(
        barmode="overlay", title=title,
        xaxis_title=col, yaxis_title="Count",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#e2e8f0",
    )
    return fig


# ── Tab renderers ─────────────────────────────────────────────────────────────

def tab_overview(df: pd.DataFrame):
    st.markdown('<div class="section-header">📊 Dataset Overview</div>',
                unsafe_allow_html=True)

    total = len(df)
    n_sat = (df["satisfaction"] == "satisfied").sum()
    n_dis = total - n_sat

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, f"{total:,}", "Total Passengers"),
        (c2, f"{df.shape[1]-1}", "Features"),
        (c3, f"{n_sat/total*100:.1f}%", "Satisfied"),
        (c4, f"{n_dis/total*100:.1f}%", "Neutral/Dissatisfied"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        counts = df["satisfaction"].value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     color=counts.index,
                     color_discrete_map={"satisfied": COLOR_SAT,
                                         "neutral or dissatisfied": COLOR_DISSAT},
                     title="Target Distribution")
        fig.update_layout(paper_bgcolor="#0d1117", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🔑 Key Findings")
        findings = [
            "Business class passengers show significantly higher satisfaction rates.",
            "Business travel purposes correlate with higher satisfaction than personal travel.",
            "Online boarding rating is among the strongest predictors of satisfaction.",
            "Loyal customers are more satisfied than disloyal customers.",
            "Inflight entertainment and seat comfort strongly associate with satisfaction.",
            "Long departure delays (>45 min) correlate with lower satisfaction.",
        ]
        for f in findings:
            st.markdown(f'<div class="insight-box">💡 {f}</div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Dataset Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)


def tab_passenger_analysis(df: pd.DataFrame):
    st.markdown('<div class="section-header">👤 Passenger Demographics & Travel</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plotly_bar_satisfaction(df, "Gender",
                        "Satisfaction Rate by Gender"), use_container_width=True)
        st.plotly_chart(plotly_bar_satisfaction(df, "Customer Type",
                        "Satisfaction Rate by Customer Type"), use_container_width=True)
    with c2:
        st.plotly_chart(plotly_bar_satisfaction(df, "Type of Travel",
                        "Satisfaction Rate by Type of Travel"), use_container_width=True)
        st.plotly_chart(plotly_bar_satisfaction(df, "Class",
                        "Satisfaction Rate by Travel Class"), use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plotly_histogram(df, "Age",
                        "Age Distribution by Satisfaction"), use_container_width=True)
    with c2:
        df2 = df.copy()
        df2["Distance Bin"] = pd.cut(df2["Flight Distance"],
                                      bins=[0, 500, 1000, 2000, 5000],
                                      labels=["<500", "500-1000", "1000-2000", ">2000"])
        st.plotly_chart(plotly_bar_satisfaction(df2, "Distance Bin",
                        "Satisfaction by Flight Distance (miles)"), use_container_width=True)

    # Class × Travel type heatmap
    pivot = (df.groupby(["Class", "Type of Travel"])["satisfaction"]
               .apply(lambda s: (s == "satisfied").mean() * 100)
               .unstack())
    fig = px.imshow(pivot, text_auto=".1f", color_continuous_scale="RdYlGn",
                    zmin=0, zmax=100,
                    title="Satisfaction Rate (%) — Class × Travel Type")
    fig.update_layout(paper_bgcolor="#0d1117", font_color="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)


def tab_service_quality(df: pd.DataFrame):
    st.markdown('<div class="section-header">⭐ Service Quality Analysis</div>',
                unsafe_allow_html=True)

    RATING_COLS = [
        "Inflight wifi service", "Departure/Arrival time convenient",
        "Ease of Online booking", "Gate location", "Food and drink",
        "Online boarding", "Seat comfort", "Inflight entertainment",
        "On-board service", "Leg room service", "Baggage handling",
        "Checkin service", "Inflight service", "Cleanliness",
    ]

    means = df.groupby("satisfaction")[RATING_COLS].mean()
    fig = go.Figure()
    for sat in means.index:
        color = COLOR_SAT if sat == "satisfied" else COLOR_DISSAT
        fig.add_trace(go.Bar(
            name=sat,
            x=[c[:20] for c in RATING_COLS],
            y=means.loc[sat].values,
            marker_color=color,
        ))
    fig.update_layout(
        barmode="group",
        title="Mean Service Ratings by Satisfaction Group",
        xaxis_title="Service Feature",
        yaxis_title="Mean Rating (0–5)",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#e2e8f0",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    selected_service = st.selectbox("Explore individual service feature:",
                                     RATING_COLS)
    st.plotly_chart(plotly_bar_satisfaction(df, selected_service,
                    f"Satisfaction Rate by {selected_service} Rating"),
                    use_container_width=True)

    # Difference chart
    diff = means.loc["satisfied"] - means.loc["neutral or dissatisfied"]
    diff_df = diff.reset_index()
    diff_df.columns = ["Feature", "Difference"]
    diff_df = diff_df.sort_values("Difference", ascending=True)
    fig2 = px.bar(diff_df, x="Difference", y="Feature", orientation="h",
                  color="Difference", color_continuous_scale="RdYlGn",
                  title="Mean Rating Difference: Satisfied − Neutral/Dissatisfied",
                  labels={"Difference": "Rating Difference"})
    fig2.update_layout(paper_bgcolor="#0d1117", font_color="#e2e8f0")
    st.plotly_chart(fig2, use_container_width=True)

    st.info("⚠️ These are observational associations, not causal relationships. "
            "Higher ratings in satisfied passengers do not necessarily mean "
            "improving those services will cause satisfaction to increase.")


def tab_delay_analysis(df: pd.DataFrame):
    st.markdown('<div class="section-header">⏱️ Flight Delay Analysis</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plotly_histogram(df.assign(**{"Departure Delay (clipped)":
                             df["Departure Delay in Minutes"].clip(upper=df["Departure Delay in Minutes"].quantile(0.99))}),
                             "Departure Delay (clipped)", "Departure Delay Distribution (≤99th pct)"),
            use_container_width=True)
    with c2:
        arr = df["Arrival Delay in Minutes"].clip(
            upper=df["Arrival Delay in Minutes"].quantile(0.99))
        df2 = df.copy()
        df2["Arrival Delay (clipped)"] = arr
        st.plotly_chart(
            plotly_histogram(df2, "Arrival Delay (clipped)",
                             "Arrival Delay Distribution (≤99th pct)"),
            use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    for col_name, label in [
        ("Departure Delay in Minutes", "Departure"),
        ("Arrival Delay in Minutes",   "Arrival"),
    ]:
        df3 = df.copy()
        df3["Delay Bin"] = pd.cut(
            df3[col_name],
            bins=[-1, 0, 15, 45, 120, df3[col_name].max() + 1],
            labels=["None", "1-15", "16-45", "46-120", ">120"]
        )
        with (c1 if label == "Departure" else c2):
            st.plotly_chart(plotly_bar_satisfaction(df3, "Delay Bin",
                            f"Satisfaction by {label} Delay Bucket"),
                            use_container_width=True)

    # Scatter
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="Departure Delay in Minutes",
        y="Arrival Delay in Minutes",
        color="satisfaction",
        color_discrete_map=PALETTE,
        opacity=0.4,
        title="Departure vs Arrival Delay (sample)",
        range_x=[0, sample["Departure Delay in Minutes"].quantile(0.99)],
        range_y=[0, sample["Arrival Delay in Minutes"].quantile(0.99)],
    )
    fig.update_layout(paper_bgcolor="#0d1117", font_color="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)


def tab_model_performance():
    st.markdown('<div class="section-header">🤖 Model Performance</div>',
                unsafe_allow_html=True)

    comparison = load_comparison()
    metrics = load_metrics()
    history = load_mlp_history()

    if comparison is None or metrics is None:
        st.warning("Model metrics not found. Run `python -m src.evaluate` first.")
        return

    st.markdown("### 📊 Model Comparison Table")
    styled = comparison.style.format({
        c: "{:.4f}" for c in comparison.columns if c != "model"
    }).background_gradient(cmap="RdYlGn", subset=[
        "accuracy", "precision", "recall", "f1_score", "roc_auc"
    ])
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    metric_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    fig = go.Figure()
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    for i, row in comparison.iterrows():
        fig.add_trace(go.Bar(
            name=row["model"],
            x=metric_cols,
            y=[row[m] for m in metric_cols],
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        barmode="group",
        title="Model Metrics Comparison",
        yaxis_range=[0.75, 1.01],
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#e2e8f0",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ROC and confusion matrix images
    col1, col2 = st.columns(2)
    roc_path = FIGURES_DIR / "eval_roc_curves.png"
    pr_path  = FIGURES_DIR / "eval_pr_curves.png"
    cm_path  = FIGURES_DIR / "eval_confusion_matrices.png"

    with col1:
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Curves — Test Set")
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrices — Test Set")
    with col2:
        if pr_path.exists():
            st.image(str(pr_path), caption="Precision-Recall Curves — Test Set")

    # MLP training history
    if history:
        st.markdown("### 🧠 MLP Training History")
        fig2 = make_subplots(rows=1, cols=2,
                             subplot_titles=("Loss", "Accuracy"))
        for curve, color in [("loss", "#e74c3c"), ("val_loss", "#3498db")]:
            if curve in history:
                fig2.add_trace(go.Scatter(y=history[curve], name=curve,
                                          line={"color": color}), row=1, col=1)
        for curve, color in [("accuracy", "#e74c3c"), ("val_accuracy", "#3498db")]:
            if curve in history:
                fig2.add_trace(go.Scatter(y=history[curve], name=curve,
                                          line={"color": color}), row=1, col=2)
        fig2.update_layout(paper_bgcolor="#0d1117", font_color="#e2e8f0",
                           title="MLP Training History")
        st.plotly_chart(fig2, use_container_width=True)


def tab_prediction(artifacts: dict):
    st.markdown('<div class="section-header">🔮 Passenger Satisfaction Prediction</div>',
                unsafe_allow_html=True)

    if "pipeline" not in artifacts:
        st.warning("Models not found. Run `python -m src.train` first.")
        return

    RATING_COLS = [
        "Inflight wifi service", "Departure/Arrival time convenient",
        "Ease of Online booking", "Gate location", "Food and drink",
        "Online boarding", "Seat comfort", "Inflight entertainment",
        "On-board service", "Leg room service", "Baggage handling",
        "Checkin service", "Inflight service", "Cleanliness",
    ]

    st.markdown("Fill in passenger details below to get a satisfaction prediction.")
    st.info("⚠️ This is a **model estimate** based on training data patterns, "
            "not a guarantee of actual passenger experience.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender        = st.selectbox("Gender", ["Male", "Female"])
            customer_type = st.selectbox("Customer Type",
                                          ["Loyal Customer", "disloyal Customer"])
            age           = st.number_input("Age", min_value=1, max_value=100, value=35)
        with c2:
            travel_type   = st.selectbox("Type of Travel",
                                          ["Business travel", "Personal Travel"])
            travel_class  = st.selectbox("Class", ["Business", "Eco Plus", "Eco"])
            flight_dist   = st.number_input("Flight Distance (miles)",
                                             min_value=1, max_value=6000, value=1000)
        with c3:
            dep_delay = st.number_input("Departure Delay (minutes)",
                                         min_value=0, max_value=1500, value=0)
            arr_delay = st.number_input("Arrival Delay (minutes)",
                                         min_value=0, max_value=1500, value=0)
            model_choice = st.selectbox("Model",
                                         [k for k in ["hgb", "random_forest",
                                                       "logistic_regression", "mlp"]
                                          if k in artifacts],
                                         format_func=lambda x: {
                                             "hgb":  "HistGradientBoosting (recommended)",
                                             "random_forest": "Random Forest",
                                             "logistic_regression": "Logistic Regression",
                                             "mlp": "Neural Network (MLP)",
                                         }.get(x, x))

        st.markdown("#### Service Ratings (0 = Not Applicable, 1–5 = Low to High)")
        rating_cols_mid = len(RATING_COLS) // 2
        rc1, rc2 = st.columns(2)
        ratings = {}
        for i, col in enumerate(RATING_COLS):
            with (rc1 if i < rating_cols_mid else rc2):
                ratings[col] = st.slider(col, 0, 5, 3)

        submitted = st.form_submit_button("🔮 Predict Satisfaction", type="primary")

    if submitted:
        record = {
            "Gender":                            gender,
            "Customer Type":                     customer_type,
            "Age":                               age,
            "Type of Travel":                    travel_type,
            "Class":                             travel_class,
            "Flight Distance":                   flight_dist,
            "Departure Delay in Minutes":        dep_delay,
            "Arrival Delay in Minutes":          float(arr_delay),
            **ratings,
        }
        try:
            from src.predict import predict_single
            result = predict_single(record, model_name=model_choice)

            pred_class = result["predicted_class"]
            p_sat = result["probability_satisfied"]
            p_dis = result["probability_dissatisfied"]

            if pred_class == "satisfied":
                st.markdown(f"""<div class="pred-satisfied">
                    <h2>✅ Predicted: SATISFIED</h2>
                    <p style="font-size:1.2rem">Confidence: <strong>{p_sat*100:.1f}%</strong></p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="pred-dissatisfied">
                    <h2>❌ Predicted: NEUTRAL / DISSATISFIED</h2>
                    <p style="font-size:1.2rem">Confidence: <strong>{p_dis*100:.1f}%</strong></p>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(
                    x=["Satisfied", "Neutral / Dissatisfied"],
                    y=[p_sat, p_dis],
                    marker_color=[COLOR_SAT, COLOR_DISSAT],
                ))
                fig.update_layout(
                    title="Predicted Probabilities",
                    yaxis_title="Probability",
                    yaxis_range=[0, 1],
                    plot_bgcolor="#0d1117",
                    paper_bgcolor="#0d1117",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("**Input Summary**")
                st.json({k: v for k, v in record.items()})

        except (ValueError, KeyError, TypeError, RuntimeError, AttributeError) as e:
            st.error(f"Prediction failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <h1 style="font-size:2.5rem; font-weight:700; color:#e2e8f0;">
            ✈️ Airline Passenger Satisfaction
        </h1>
        <p style="color:#a0aec0; font-size:1rem;">
            End-to-End Analysis, ML Modeling & Interactive Prediction Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)

    df_full = load_data()
    artifacts = load_artifacts()
    df = render_sidebar(df_full)

    tabs = st.tabs([
        "📊 Overview",
        "👤 Passenger Analysis",
        "⭐ Service Quality",
        "⏱️ Delay Analysis",
        "🤖 Model Performance",
        "🔮 Predict",
    ])

    with tabs[0]:
        tab_overview(df)
    with tabs[1]:
        tab_passenger_analysis(df)
    with tabs[2]:
        tab_service_quality(df)
    with tabs[3]:
        tab_delay_analysis(df)
    with tabs[4]:
        tab_model_performance()
    with tabs[5]:
        tab_prediction(artifacts)


if __name__ == "__main__":
    main()

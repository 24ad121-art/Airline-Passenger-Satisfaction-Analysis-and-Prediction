# ✈️ Airline Passenger Satisfaction Analysis & Prediction

An end-to-end, reproducible Data Science, Machine Learning, and Deep Learning system designed to analyze airline passenger satisfaction drivers and accurately predict passenger sentiment.

Built for academic submission, undergraduate portfolio showcases, and production-ready machine learning pipelines.

---

## 📌 Executive Summary

Passenger satisfaction directly impacts retention, loyalty, airline brand reputation, and operating margins. This project provides a complete machine learning lifecycle solution analyzing **129,880 passenger records** (103,904 train / 25,976 test) across 22 operational, demographic, and flight-service dimensions.

### Key Highlights
* **Strict Leakage Prevention**: Pipeline strictly fit on training splits; IDs (`Unnamed: 0`, `id`) and target labels dropped before feature transformation. Median delay imputation and ordinal encoding with graceful unknown handling.
* **Rigorous Modeling Benchmark**: 4 distinct architectures evaluated on held-out test data — **Logistic Regression (Baseline)**, **Random Forest**, **Histogram Gradient Boosting (HGB)**, and a **Deep Learning Multi-Layer Perceptron (MLP)** with Batch Normalization and Dropout regularization.
* **Top Performance**: Deep Learning MLP achieves **96.50% Accuracy** and **0.9953 ROC-AUC**; Gradient Boosting achieves **96.47% Accuracy** and **0.9953 ROC-AUC**.
* **Model Explainability & XAI**: Global and local feature attribution using **SHAP (SHapley Additive exPlanations)** beeswarm and importance plots, revealing the critical role of *Inflight WiFi*, *Online Boarding*, *Type of Travel*, and *Class*.
* **Interactive Web Application**: Streamlit dashboard with 5 specialized modules: Executive Overview, Exploratory Analytics, Model Comparison & Curves, Real-Time Prediction Simulator, and SHAP Explainability Explorer.
* **Production-Grade Engineering**: 46 automated unit tests (`pytest`), reproducible seed controls, modular architecture (`src/`), and automated artifact generation.

---

## 📊 Dataset Overview

* **Training Set**: 103,904 rows × 25 columns
* **Held-out Test Set**: 25,976 rows × 25 columns
* **Target Variable**: `satisfaction`
  * `neutral or dissatisfied` (Class 0 — ~56.7% in train)
  * `satisfied` (Class 1 — ~43.3% in train)
* **Feature Categories**:
  * **Demographics & Travel**: Gender, Customer Type (Loyal vs. Disloyal), Age, Type of Travel (Business vs. Personal), Class (Business, Eco, Eco Plus), Flight Distance.
  * **In-Flight & Ground Services (Ratings 0–5)**: Inflight wifi service, Departure/Arrival time convenient, Ease of Online booking, Gate location, Food and drink, Online boarding, Seat comfort, Inflight entertainment, On-board service, Leg room service, Baggage handling, Checkin service, Inflight service, Cleanliness.
  * **Flight Operational Delays**: Departure Delay in Minutes, Arrival Delay in Minutes (median imputed).

---

## 🏆 Model Performance Benchmark (Held-out Test Set)

Evaluated on $N = 25,976$ strictly held-out test records:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Linear Baseline) | 87.10% | 86.76% | 83.33% | 0.8501 | 0.9253 | 0.9304 |
| **Random Forest** (300 Trees, Min Leaf 2) | 96.17% | 97.01% | 94.17% | 0.9557 | 0.9938 | 0.9930 |
| **HistGradientBoosting (HGB)** | 96.47% | **97.42%** | 94.45% | 0.9591 | **0.9953** | 0.9945 |
| **Deep Learning MLP** (Keras / TensorFlow) | **96.50%** | 97.31% | **94.65%** | **0.9596** | **0.9953** | **0.9946** |

### Evaluation Curves & Artifacts
The pipeline automatically generates high-resolution evaluation figures in `reports/figures/`:
* `eval_roc_curves.png`: Multi-model ROC curves showing near-perfect separation (AUC > 0.99) for ensemble and deep learning models.
* `eval_pr_curves.png`: Precision-Recall curves maintaining > 0.97 precision even at high recall.
* `eval_confusion_matrices.png`: Side-by-side confusion matrices verifying low false-negative and false-positive rates.
* `eval_mlp_history.png`: Training vs. Validation loss and accuracy curves illustrating stable convergence with early stopping.
* `eval_model_comparison.png`: Comprehensive bar chart comparing all 5 primary metrics.

---

## 🔍 Key Insights & Explainability (SHAP)

Using TreeSHAP and Mean Absolute SHAP values on 22 features, the top satisfaction drivers are:

1. **Type of Travel (Business vs Personal)** ($\text{SHAP} \approx 1.96$): Business travelers exhibit significantly higher baseline expectations but report higher satisfaction when core digital services perform well. Personal travelers fly predominantly in Economy and show higher sensitivity to price and delays.
2. **Inflight WiFi Service** ($\text{SHAP} \approx 1.67$): A strong non-linear tipping point. Ratings $\ge 4$ correlate with an 85%+ satisfaction probability, whereas ratings $\le 2$ virtually guarantee passenger dissatisfaction regardless of seat quality.
3. **Online Boarding** ($\text{SHAP} \approx 1.60$): The single most impactful pre-flight touchpoint. Smooth digital check-in directly correlates with positive post-flight sentiment.
4. **Customer Type (Loyal vs Disloyal)** ($\text{SHAP} \approx 0.82$): Disloyal/first-time passengers exhibit much lower tolerance for service friction.
5. **Travel Class (Business vs Economy)** ($\text{SHAP} \approx 0.54$): Business class passengers experience higher legroom, seat comfort, and personalized flight attendant attention.

---

## 🗂️ Project Structure

```text
Airline Passenger Satisfaction Analysis and Prediction/
├── dataset/
│   ├── train.csv                      # Primary training dataset (103,904 rows)
│   └── test.csv                       # Held-out evaluation dataset (25,976 rows)
├── models/
│   ├── preprocessing_pipeline.pkl     # Fitted Scikit-Learn transformer pipeline
│   ├── logistic_regression.pkl        # Baseline linear classifier
│   ├── random_forest.pkl              # 300-estimator Random Forest classifier
│   ├── hgb.pkl                        # HistGradientBoosting model (champion)
│   ├── mlp_best.keras                 # Best checkpointed Deep Learning Keras MLP
│   ├── mlp_history.json               # Keras training/validation trajectory
│   └── val_predictions.npy            # Saved validation probability predictions
├── notebooks/
│   └── airline_satisfaction_eda.ipynb # Reproducible exploratory data analysis notebook
├── reports/
│   ├── figures/                       # Generated EDA & Evaluation high-res charts (19+ PNGs)
│   ├── dataset_summary.json           # Statistical summary of distributions and nulls
│   ├── model_comparison.csv           # Model benchmark metrics table
│   ├── model_metrics.json             # Full classification reports and metrics
│   └── shap_importance.json           # Top global SHAP feature importances
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Schema validation, schema constants, load helpers
│   ├── preprocessing.py               # Leak-free sklearn pipeline construction & transforms
│   ├── eda.py                         # Automated statistical analysis & figure generation
│   ├── train.py                       # Training pipeline for Sklearn + Keras models
│   ├── evaluate.py                    # Test-set evaluation, metric computation, ROC/PR plots
│   ├── explainability.py              # SHAP values, beeswarm plots, and MDI importance
│   └── predict.py                     # Real-time single & batch prediction engine
├── tests/
│   ├── test_data_loader.py            # Unit tests for loading, schemas, and encoding
│   ├── test_preprocessing.py          # Unit tests for imputation, pipeline fit, unseen values
│   └── test_prediction.py             # Unit tests for inference probability bounds & shapes
├── app.py                             # Interactive Streamlit dashboard
├── conftest.py                        # Pytest configuration
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Environment Setup
Ensure Python 3.10+ is installed:
```bash
git clone https://github.com/your-username/airline-passenger-satisfaction.git
cd "Airline Passenger Satisfaction Analysis and Prediction"

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis (EDA)
Generates 14+ statistical visualization figures and summary statistics in `reports/`:
```bash
python run_eda.py
```

### 3. Train Models
Fits the preprocessing pipeline on training data, trains all 4 models, and serializes them to `models/`:
```bash
python -m src.train
```

### 4. Evaluate Models on Held-Out Test Set
Evaluates models against `dataset/test.csv`, writes `reports/model_comparison.csv`, and renders ROC, PR, and Confusion Matrix charts:
```bash
python -m src.evaluate
```

### 5. Run SHAP Explainability Analysis
Computes TreeSHAP feature attributions and generates beeswarm & bar plots:
```bash
python -m src.explainability
```

### 6. Run Unit Tests & Verify Coverage
Executes the comprehensive automated test suite (46 tests):
```bash
pytest -v --cov=src
```

### 7. Launch the Interactive Dashboard
Starts the multi-page Streamlit web application:
```bash
streamlit run app.py
```

---

## 💻 Interactive Dashboard Features

The Streamlit web application (`app.py`) includes:

1. **Executive Overview**: High-level KPI metric cards, target class distributions, and summary insights.
2. **Exploratory Data Analysis**: Interactive Plotly visualizations exploring age distributions, travel type breakdown, service rating correlations, and delay impacts.
3. **Model Performance & Benchmarking**: Interactive comparison tables, ROC curves, PR curves, and side-by-side confusion matrix inspections.
4. **Real-Time Prediction Simulator**: Dynamic sliders and dropdowns allowing users to configure passenger characteristics and flight ratings to instantly receive prediction probabilities, sentiment gauge, and personalized satisfaction recommendations.
5. **SHAP Explainability Explorer**: Visual explanation of which features drive satisfaction or dissatisfaction, backed by TreeSHAP analysis.

---

## 🛡️ Best Practices & Architecture Highlights

* **No Data Leakage**: Transformations are computed solely on training subsets. Missing value imputers and standard scalers never observe test data during fitting.
* **Deterministic Sentinels for Unseen Categories**: Ordinal encoders utilize `handle_unknown='use_encoded_value'` with `unknown_value=-1` to ensure inference stability on edge cases.
* **Reproducibility**: Explicit random seed control (`SEED = 42`) across Python, NumPy, Scikit-Learn, and TensorFlow.
* **Modularity**: Separation of concerns between ingestion (`data_loader`), transformation (`preprocessing`), modeling (`train`), evaluation (`evaluate`), and deployment (`app.py`).

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

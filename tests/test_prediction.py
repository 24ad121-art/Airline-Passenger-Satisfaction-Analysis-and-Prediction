"""
Tests for predict.py
"""
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest


MODELS_DIR = Path("models")


def models_available() -> bool:
    return (
        (MODELS_DIR / "preprocessing_pipeline.pkl").exists()
        and (MODELS_DIR / "hgb.pkl").exists()
    )


@pytest.mark.skipif(not models_available(), reason="Models not yet trained")
class TestPredictSingle:
    """Test single-record prediction."""

    EXAMPLE_RECORD: ClassVar[dict] = {
        "Gender": "Female",
        "Customer Type": "Loyal Customer",
        "Age": 42,
        "Type of Travel": "Business travel",
        "Class": "Business",
        "Flight Distance": 1500,
        "Inflight wifi service": 4,
        "Departure/Arrival time convenient": 3,
        "Ease of Online booking": 4,
        "Gate location": 3,
        "Food and drink": 4,
        "Online boarding": 5,
        "Seat comfort": 5,
        "Inflight entertainment": 5,
        "On-board service": 4,
        "Leg room service": 4,
        "Baggage handling": 5,
        "Checkin service": 4,
        "Inflight service": 5,
        "Cleanliness": 5,
        "Departure Delay in Minutes": 0,
        "Arrival Delay in Minutes": 0,
    }

    def test_prediction_returns_dict(self):
        from src.predict import predict_single
        result = predict_single(self.EXAMPLE_RECORD)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        from src.predict import predict_single
        result = predict_single(self.EXAMPLE_RECORD)
        for key in ["predicted_class", "probability_satisfied", "probability_dissatisfied"]:
            assert key in result

    def test_probability_sum_to_one(self):
        from src.predict import predict_single
        result = predict_single(self.EXAMPLE_RECORD)
        total = result["probability_satisfied"] + result["probability_dissatisfied"]
        assert abs(total - 1.0) < 1e-6

    def test_probability_in_range(self):
        from src.predict import predict_single
        result = predict_single(self.EXAMPLE_RECORD)
        assert 0.0 <= result["probability_satisfied"] <= 1.0
        assert 0.0 <= result["probability_dissatisfied"] <= 1.0

    def test_predicted_class_valid(self):
        from src.predict import predict_single
        result = predict_single(self.EXAMPLE_RECORD)
        assert result["predicted_class"] in {"satisfied", "neutral or dissatisfied"}

    def test_missing_values_handled(self):
        """Prediction with missing (NaN) fields should still work."""
        from src.predict import predict_single
        partial_record = {"Gender": "Male", "Age": 30}
        result = predict_single(partial_record)
        assert 0.0 <= result["probability_satisfied"] <= 1.0


@pytest.mark.skipif(not models_available(), reason="Models not yet trained")
class TestPredictBatch:
    def test_batch_output_shape(self):
        from src.data_loader import get_feature_matrix, load_csv
        from src.predict import predict_batch

        df = load_csv("dataset/test.csv")
        X = get_feature_matrix(df).head(50)
        probs = predict_batch(X)
        assert probs.shape == (50,)

    def test_batch_probabilities_valid(self):
        from src.data_loader import get_feature_matrix, load_csv
        from src.predict import predict_batch
        df = load_csv("dataset/test.csv")
        X = get_feature_matrix(df).head(50)
        probs = predict_batch(X)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)


@pytest.mark.skipif(not models_available(), reason="Models not yet trained")
class TestModelArtifacts:
    def test_pipeline_loads(self):
        from src.preprocessing import load_pipeline
        pipeline = load_pipeline()
        assert pipeline is not None

    def test_hgb_loads(self):
        from src.train import load_model
        model = load_model("hgb")
        assert model is not None

    def test_logistic_regression_loads(self):
        from src.train import load_model
        model = load_model("logistic_regression")
        assert model is not None

    def test_random_forest_loads(self):
        from src.train import load_model
        model = load_model("random_forest")
        assert model is not None

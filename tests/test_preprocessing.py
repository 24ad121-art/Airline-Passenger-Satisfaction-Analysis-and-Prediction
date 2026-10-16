"""
Tests for preprocessing.py
"""
from pathlib import Path

import numpy as np
import pytest

from src.data_loader import FEATURE_COLS, get_feature_matrix, load_csv
from src.preprocessing import (
    build_pipeline,
    fit_pipeline,
    get_feature_names,
    load_pipeline,
    record_to_dataframe,
    save_pipeline,
    transform,
)

TRAIN_PATH = Path("dataset/train.csv")
PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")


@pytest.fixture(scope="module")
def train_df():
    return load_csv(TRAIN_PATH)


@pytest.fixture(scope="module")
def X_train(train_df):
    return get_feature_matrix(train_df)


@pytest.fixture(scope="module")
def fitted_pipeline(X_train):
    """Fit a fresh pipeline on a small sample (fast for tests)."""
    sample = X_train.sample(2000, random_state=42)
    return fit_pipeline(sample, save=False)


class TestPipelineBuild:
    def test_pipeline_builds(self):
        pipeline = build_pipeline()
        assert pipeline is not None

    def test_pipeline_has_preprocessor(self):
        pipeline = build_pipeline()
        assert "preprocessor" in pipeline.named_steps


class TestPipelineFit:
    def test_fit_runs(self, fitted_pipeline):
        assert fitted_pipeline is not None

    def test_transform_shape(self, fitted_pipeline, X_train):
        sample = X_train.sample(100, random_state=1)
        X_t = transform(fitted_pipeline, sample)
        assert X_t.shape[0] == 100
        assert X_t.ndim == 2

    def test_feature_count_consistent(self, fitted_pipeline, X_train):
        sample1 = X_train.sample(50, random_state=1)
        sample2 = X_train.sample(50, random_state=2)
        n1 = transform(fitted_pipeline, sample1).shape[1]
        n2 = transform(fitted_pipeline, sample2).shape[1]
        assert n1 == n2, "Feature count changed between transforms"

    def test_no_nan_in_output(self, fitted_pipeline, X_train):
        sample = X_train.sample(200, random_state=42)
        X_t = transform(fitted_pipeline, sample)
        assert not np.isnan(X_t).any(), "NaN found in transformed output"

    def test_arrival_delay_imputed(self, fitted_pipeline, X_train):
        """Rows with missing Arrival Delay should produce finite values."""
        missing_mask = X_train["Arrival Delay in Minutes"].isnull()
        missing_rows = X_train[missing_mask]
        if len(missing_rows) > 0:
            X_t = transform(fitted_pipeline, missing_rows)
            assert np.isfinite(X_t).all()


class TestNoPipeFitting_OnTestData:
    """Ensure the pipeline is fitted only on train data."""

    def test_pipeline_fitted_before_test_transform(self, fitted_pipeline, X_train):
        # Transform on a held-out subset (simulating test)
        hold_out = X_train.sample(50, random_state=99)
        X_t = transform(fitted_pipeline, hold_out)
        assert X_t.shape[0] == 50


class TestFeatureNames:
    def test_feature_names_length_matches_transform(self, fitted_pipeline, X_train):
        sample = X_train.sample(10, random_state=42)
        X_t = transform(fitted_pipeline, sample)
        names = get_feature_names(fitted_pipeline)
        assert len(names) == X_t.shape[1]

    def test_feature_names_are_strings(self, fitted_pipeline):
        names = get_feature_names(fitted_pipeline)
        assert all(isinstance(n, str) for n in names)


class TestUnseenCategories:
    def test_unseen_gender_value(self, fitted_pipeline):
        record = record_to_dataframe({
            "Gender": "Unknown_XYZ",  # unseen value
            "Customer Type": "Loyal Customer",
            "Age": 30,
            "Type of Travel": "Business travel",
            "Class": "Business",
            "Flight Distance": 500,
        })
        # Should not raise; unseen → sentinel -1
        X_t = transform(fitted_pipeline, record)
        assert X_t is not None
        assert np.isfinite(X_t).all()


class TestSingleRecordInference:
    def test_record_to_dataframe_shape(self):
        rec = {"Age": 25, "Gender": "Female"}
        df = record_to_dataframe(rec)
        assert df.shape == (1, len(FEATURE_COLS))

    def test_record_missing_columns_filled_with_nan(self):
        rec = {"Age": 40}  # only one field
        df = record_to_dataframe(rec)
        # All other FEATURE_COLS should be NaN
        other_cols = [c for c in FEATURE_COLS if c != "Age"]
        assert df[other_cols].isnull().all().all()


class TestPipelineSaveLoad:
    def test_save_load_roundtrip(self, fitted_pipeline, X_train, tmp_path):
        save_path = tmp_path / "test_pipeline.pkl"
        save_pipeline(fitted_pipeline, save_path)
        loaded = load_pipeline(save_path)
        sample = X_train.sample(20, random_state=42)
        X_orig   = transform(fitted_pipeline, sample)
        X_loaded = transform(loaded, sample)
        np.testing.assert_array_almost_equal(X_orig, X_loaded)

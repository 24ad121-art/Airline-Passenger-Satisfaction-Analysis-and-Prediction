"""
Tests for data_loader.py
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    FEATURE_COLS,
    ID_COLS,
    POSITIVE_CLASS,
    TARGET_COL,
    build_dataset_summary,
    get_feature_matrix,
    get_target,
    load_csv,
    validate_schema,
)

TRAIN_PATH = Path("dataset/train.csv")
TEST_PATH  = Path("dataset/test.csv")


@pytest.fixture(scope="module")
def train_df():
    return load_csv(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_df():
    return load_csv(TEST_PATH)


class TestDataLoading:
    def test_train_loads(self, train_df):
        assert isinstance(train_df, pd.DataFrame)
        assert len(train_df) > 0

    def test_test_loads(self, test_df):
        assert isinstance(test_df, pd.DataFrame)
        assert len(test_df) > 0

    def test_train_shape(self, train_df):
        rows, cols = train_df.shape
        assert rows > 100_000, f"Expected ~103904 rows, got {rows}"
        assert cols == 25, f"Expected 25 columns, got {cols}"

    def test_test_shape(self, test_df):
        rows, cols = test_df.shape
        assert rows > 25_000, f"Expected ~25976 rows, got {rows}"
        assert cols == 25, f"Expected 25 columns, got {cols}"

    def test_schema_validation(self, train_df):
        validate_schema(train_df, "train")  # should not raise

    def test_target_column_exists(self, train_df):
        assert TARGET_COL in train_df.columns

    def test_target_values(self, train_df):
        values = set(train_df[TARGET_COL].unique())
        assert values == {"satisfied", "neutral or dissatisfied"}, f"Unexpected values: {values}"

    def test_id_cols_present(self, train_df):
        for col in ID_COLS:
            assert col in train_df.columns

    def test_no_modification(self, train_df):
        """load_csv should not modify original data"""
        original_shape = train_df.shape
        _ = get_feature_matrix(train_df)
        assert train_df.shape == original_shape


class TestFeatureMatrix:
    def test_feature_matrix_excludes_ids(self, train_df):
        X = get_feature_matrix(train_df)
        for col in ID_COLS:
            assert col not in X.columns, f"ID column {col} found in feature matrix"

    def test_feature_matrix_excludes_target(self, train_df):
        X = get_feature_matrix(train_df)
        assert TARGET_COL not in X.columns

    def test_feature_cols_present(self, train_df):
        X = get_feature_matrix(train_df)
        for col in FEATURE_COLS:
            assert col in X.columns, f"Expected feature {col} missing"


class TestTargetEncoding:
    def test_binary_encoding(self, train_df):
        y = get_target(train_df)
        assert set(y.unique()).issubset({0, 1}), f"Non-binary values: {y.unique()}"

    def test_positive_class_is_one(self, train_df):
        y = get_target(train_df)
        satisfied_rows = train_df[train_df[TARGET_COL] == POSITIVE_CLASS]
        assert y.loc[satisfied_rows.index].eq(1).all()

    def test_dissatisfied_class_is_zero(self, train_df):
        y = get_target(train_df)
        dissat_rows = train_df[train_df[TARGET_COL] != POSITIVE_CLASS]
        assert y.loc[dissat_rows.index].eq(0).all()

    def test_no_nulls_in_target(self, train_df):
        y = get_target(train_df)
        assert y.isnull().sum() == 0


class TestMissingValues:
    def test_arrival_delay_has_missing(self, train_df):
        assert train_df["Arrival Delay in Minutes"].isnull().sum() > 0

    def test_other_cols_no_missing(self, train_df):
        for col in FEATURE_COLS:
            if col == "Arrival Delay in Minutes":
                continue
            n_miss = train_df[col].isnull().sum()
            assert n_miss == 0, f"Unexpected missing values in {col}: {n_miss}"


class TestDatasetSummary:
    def test_summary_structure(self, train_df, test_df):
        summary = build_dataset_summary(train_df, test_df)
        assert "train" in summary
        assert "test" in summary
        assert "feature_columns" in summary
        assert "target_column" in summary

    def test_summary_row_counts(self, train_df, test_df):
        summary = build_dataset_summary(train_df, test_df)
        assert summary["train"]["rows"] == len(train_df)
        assert summary["test"]["rows"]  == len(test_df)

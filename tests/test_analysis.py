"""Unit tests for the analysis pipeline."""

import json
import os
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import pytest

from analysis.run_analysis import (
    compute_variant_metrics,
    generate_recommendation,
    load_predictions,
    run_statistical_tests,
)


def _create_test_db(db_path: str, num_a: int = 200, num_b: int = 200) -> None:
    """Create a test database with synthetic prediction data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            request_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            model_variant TEXT NOT NULL,
            input_features TEXT NOT NULL,
            prediction REAL NOT NULL,
            prediction_probability REAL,
            latency_ms REAL NOT NULL
        )
    """)

    np.random.seed(42)
    for i in range(num_a):
        cursor.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"a-{i}",
                f"2025-01-15T10:{i // 60:02d}:{i % 60:02d}",
                "A",
                json.dumps([float(x) for x in np.random.randn(5)]),
                float(np.random.choice([0, 1], p=[0.3, 0.7])),
                round(float(np.random.uniform(0.5, 1.0)), 4),
                round(float(np.random.exponential(2.0) + 1.0), 3),
            ),
        )

    for i in range(num_b):
        cursor.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"b-{i}",
                f"2025-01-15T10:{i // 60:02d}:{i % 60:02d}",
                "B",
                json.dumps([float(x) for x in np.random.randn(5)]),
                float(np.random.choice([0, 1], p=[0.4, 0.6])),
                round(float(np.random.uniform(0.6, 1.0)), 4),
                round(float(np.random.exponential(3.0) + 1.5), 3),
            ),
        )

    conn.commit()
    conn.close()


@pytest.fixture
def test_db():
    """Create a temporary database with test data."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    _create_test_db(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def test_dataframe(test_db):
    """Load test data into a DataFrame."""
    return load_predictions(test_db)


@pytest.mark.unit
class TestLoadPredictions:
    """Tests for loading prediction data from the database."""

    def test_load_returns_dataframe(self, test_db):
        df = load_predictions(test_db)
        assert isinstance(df, pd.DataFrame)

    def test_load_correct_row_count(self, test_db):
        df = load_predictions(test_db)
        assert len(df) == 400

    def test_load_has_required_columns(self, test_db):
        df = load_predictions(test_db)
        required = {
            "request_id", "timestamp", "model_variant",
            "input_features", "prediction", "prediction_probability",
            "latency_ms",
        }
        assert required.issubset(set(df.columns))

    def test_load_nonexistent_db_raises_error(self):
        with pytest.raises(FileNotFoundError):
            load_predictions("/nonexistent/path/db.sqlite")

    def test_load_empty_db_raises_error(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE predictions (
                request_id TEXT PRIMARY KEY,
                timestamp TEXT, model_variant TEXT,
                input_features TEXT, prediction REAL,
                prediction_probability REAL, latency_ms REAL
            )
        """)
        conn.commit()
        conn.close()
        try:
            with pytest.raises(ValueError, match="No prediction records"):
                load_predictions(path)
        finally:
            os.remove(path)


@pytest.mark.unit
class TestComputeVariantMetrics:
    """Tests for computing per-variant metrics."""

    def test_returns_dict(self, test_dataframe):
        metrics = compute_variant_metrics(test_dataframe, "A")
        assert isinstance(metrics, dict)

    def test_contains_required_keys(self, test_dataframe):
        metrics = compute_variant_metrics(test_dataframe, "A")
        required_keys = [
            "count", "mean_prediction", "std_prediction",
            "mean_latency_ms", "std_latency_ms", "median_latency_ms",
            "p95_latency_ms", "p99_latency_ms", "min_latency_ms",
            "max_latency_ms", "positive_prediction_rate",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_count_is_correct(self, test_dataframe):
        metrics_a = compute_variant_metrics(test_dataframe, "A")
        metrics_b = compute_variant_metrics(test_dataframe, "B")
        assert metrics_a["count"] == 200
        assert metrics_b["count"] == 200

    def test_latency_values_are_positive(self, test_dataframe):
        metrics = compute_variant_metrics(test_dataframe, "A")
        assert metrics["mean_latency_ms"] > 0
        assert metrics["p95_latency_ms"] > 0
        assert metrics["min_latency_ms"] > 0

    def test_prediction_rate_in_range(self, test_dataframe):
        metrics = compute_variant_metrics(test_dataframe, "A")
        assert 0.0 <= metrics["positive_prediction_rate"] <= 1.0

    def test_empty_variant_returns_error(self, test_dataframe):
        metrics = compute_variant_metrics(test_dataframe, "C")
        assert "error" in metrics

    def test_p95_greater_than_median(self, test_dataframe):
        metrics = compute_variant_metrics(test_dataframe, "A")
        assert metrics["p95_latency_ms"] >= metrics["median_latency_ms"]


@pytest.mark.unit
class TestStatisticalTests:
    """Tests for statistical significance testing."""

    def test_returns_dict(self, test_dataframe):
        results = run_statistical_tests(test_dataframe)
        assert isinstance(results, dict)

    def test_contains_latency_ttest(self, test_dataframe):
        results = run_statistical_tests(test_dataframe)
        assert "latency_ttest" in results

    def test_contains_prediction_mannwhitney(self, test_dataframe):
        results = run_statistical_tests(test_dataframe)
        assert "prediction_mannwhitney" in results

    def test_contains_chi_squared(self, test_dataframe):
        results = run_statistical_tests(test_dataframe)
        assert "prediction_chi_squared" in results

    def test_ttest_has_required_fields(self, test_dataframe):
        results = run_statistical_tests(test_dataframe)
        ttest = results["latency_ttest"]
        assert "test" in ttest
        assert "p_value" in ttest
        assert "significant" in ttest
        assert "interpretation" in ttest

    def test_p_values_in_valid_range(self, test_dataframe):
        results = run_statistical_tests(test_dataframe)
        for key, result in results.items():
            if isinstance(result, dict) and "p_value" in result:
                assert 0.0 <= result["p_value"] <= 1.0, (
                    f"{key}: p_value {result['p_value']} out of range"
                )

    def test_insufficient_data_returns_error(self):
        df = pd.DataFrame({
            "model_variant": ["A"],
            "prediction": [1.0],
            "latency_ms": [5.0],
        })
        results = run_statistical_tests(df)
        assert "error" in results


@pytest.mark.unit
class TestGenerateRecommendation:
    """Tests for the recommendation engine."""

    def test_returns_dict(self):
        metrics_a = {"mean_latency_ms": 2.0, "positive_prediction_rate": 0.7}
        metrics_b = {"mean_latency_ms": 3.0, "positive_prediction_rate": 0.6}
        tests = {}
        result = generate_recommendation(metrics_a, metrics_b, tests)
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        metrics_a = {"mean_latency_ms": 2.0, "positive_prediction_rate": 0.7}
        metrics_b = {"mean_latency_ms": 3.0, "positive_prediction_rate": 0.6}
        tests = {}
        result = generate_recommendation(metrics_a, metrics_b, tests)
        assert "recommended_model" in result
        assert "reasons" in result

    def test_lower_latency_model_favored(self):
        metrics_a = {"mean_latency_ms": 1.0, "positive_prediction_rate": 0.5}
        metrics_b = {"mean_latency_ms": 5.0, "positive_prediction_rate": 0.5}
        tests = {}
        result = generate_recommendation(metrics_a, metrics_b, tests)
        assert "Model A" in result["recommended_model"]

    def test_significant_tests_add_reasons(self):
        metrics_a = {"mean_latency_ms": 2.0, "positive_prediction_rate": 0.7}
        metrics_b = {"mean_latency_ms": 3.0, "positive_prediction_rate": 0.6}
        tests = {
            "latency_ttest": {"significant": True},
            "prediction_chi_squared": {"significant": True},
        }
        result = generate_recommendation(metrics_a, metrics_b, tests)
        assert len(result["reasons"]) >= 2

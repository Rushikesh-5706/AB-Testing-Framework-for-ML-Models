"""Unit tests for A/B testing API endpoints."""

import sqlite3

import pytest


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, test_client):
        response = test_client.get("/health")
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "database_connected" in data
        assert "experiment_config" in data

    def test_health_models_loaded(self, test_client):
        response = test_client.get("/health")
        data = response.json()
        assert data["models_loaded"]["A"] is True
        assert data["models_loaded"]["B"] is True

    def test_health_database_connected(self, test_client):
        response = test_client.get("/health")
        data = response.json()
        assert data["database_connected"] is True


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_returns_200(self, test_client, sample_features):
        response = test_client.post("/predict", json={"features": sample_features})
        assert response.status_code == 200

    def test_predict_response_contains_required_fields(self, test_client, sample_features):
        response = test_client.post("/predict", json={"features": sample_features})
        data = response.json()
        assert "request_id" in data
        assert "model_variant" in data
        assert "prediction" in data
        assert "latency_ms" in data
        assert "prediction_probability" in data

    def test_predict_model_variant_is_valid(self, test_client, sample_features):
        response = test_client.post("/predict", json={"features": sample_features})
        data = response.json()
        assert data["model_variant"] in ("A", "B")

    def test_predict_latency_is_positive(self, test_client, sample_features):
        response = test_client.post("/predict", json={"features": sample_features})
        data = response.json()
        assert data["latency_ms"] > 0

    def test_predict_logs_to_database(self, test_client, sample_features, temp_db_path):
        response = test_client.post("/predict", json={"features": sample_features})
        data = response.json()

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM predictions WHERE request_id = ?",
            (data["request_id"],),
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[2] == data["model_variant"]

    def test_predict_with_session_id(self, test_client, sample_features):
        response = test_client.post(
            "/predict",
            json={"features": sample_features, "session_id": "test-session-123"},
        )
        assert response.status_code == 200

    def test_predict_empty_features_rejected(self, test_client):
        response = test_client.post("/predict", json={"features": []})
        assert response.status_code == 422

    def test_predict_missing_features_rejected(self, test_client):
        response = test_client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_invalid_feature_type_rejected(self, test_client):
        response = test_client.post("/predict", json={"features": ["not", "numbers"]})
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint."""

    def test_batch_predict_returns_200(self, test_client, sample_features):
        payload = {
            "requests": [
                {"features": sample_features},
                {"features": sample_features},
            ]
        }
        response = test_client.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_predict_returns_correct_count(self, test_client, sample_features):
        payload = {
            "requests": [{"features": sample_features} for _ in range(5)]
        }
        response = test_client.post("/predict/batch", json=payload)
        data = response.json()
        assert len(data["predictions"]) == 5

    def test_batch_predict_contains_total_latency(self, test_client, sample_features):
        payload = {"requests": [{"features": sample_features}]}
        response = test_client.post("/predict/batch", json=payload)
        data = response.json()
        assert "total_latency_ms" in data
        assert data["total_latency_ms"] > 0


class TestExperimentConfigEndpoint:
    """Tests for the /experiment/config endpoints."""

    def test_get_config_returns_200(self, test_client):
        response = test_client.get("/experiment/config")
        assert response.status_code == 200

    def test_get_config_response_structure(self, test_client):
        response = test_client.get("/experiment/config")
        data = response.json()
        assert "model_a_ratio" in data
        assert "model_b_ratio" in data
        assert "model_a_name" in data
        assert "model_b_name" in data

    def test_update_config(self, test_client):
        response = test_client.post(
            "/experiment/config",
            json={"model_a_ratio": 0.7},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_a_ratio"] == 0.7
        assert abs(data["model_b_ratio"] - 0.3) < 1e-9

    def test_update_config_invalid_ratio_rejected(self, test_client):
        response = test_client.post(
            "/experiment/config",
            json={"model_a_ratio": 1.5},
        )
        assert response.status_code == 422

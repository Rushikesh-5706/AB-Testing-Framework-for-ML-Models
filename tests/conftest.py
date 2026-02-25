"""Shared test fixtures for the A/B testing framework test suite."""

import asyncio
import os
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient


class MockModel:
    """Mock ML model that mimics scikit-learn's predict/predict_proba API."""

    def __init__(self, return_value: float = 1.0, proba: float = 0.85) -> None:
        self._return_value = return_value
        self._proba = proba

    def predict(self, X) -> np.ndarray:
        return np.array([self._return_value] * len(X))

    def predict_proba(self, X) -> np.ndarray:
        return np.array([[1 - self._proba, self._proba]] * len(X))


@pytest.fixture
def mock_models() -> dict:
    """Provide mock models for testing without loading real pickle files."""
    return {
        "A": MockModel(return_value=1.0, proba=0.85),
        "B": MockModel(return_value=0.0, proba=0.72),
    }


@pytest.fixture
def temp_db_path():
    """Provide a temporary database file path for isolated testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def test_client(mock_models, temp_db_path):
    """Create a FastAPI test client with mock models and temporary database."""
    import api.main as main_module
    from api.traffic_splitter import TrafficSplitter

    original_models = main_module.models.copy()
    original_db_path = main_module.db.db_path
    original_connection = main_module.db._connection

    main_module.models.update(mock_models)
    main_module.db.db_path = temp_db_path
    main_module.db._connection = None  # force lifespan to initialize fresh
    main_module.splitter.update_ratio(0.5)  # reset to default before each test

    with TestClient(main_module.app) as client:
        yield client

    main_module.splitter.update_ratio(0.5)  # reset after test too
    main_module.models.clear()
    main_module.models.update(original_models)
    main_module.db.db_path = original_db_path
    main_module.db._connection = original_connection


@pytest.fixture
def sample_features() -> list[float]:
    """Provide sample features matching Breast Cancer dataset dimensions (30 features)."""
    return [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
        0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
        0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
        0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
    ]

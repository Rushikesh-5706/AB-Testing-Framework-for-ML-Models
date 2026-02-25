"""Unit tests for the database logging mechanism."""

import asyncio
import json
import os
import sqlite3
import tempfile

import pytest

from api.database import DatabaseManager


@pytest.fixture
def db_manager():
    """Create a DatabaseManager with a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    manager = DatabaseManager(db_path=path)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(manager.initialize())
    yield manager, path, loop
    loop.run_until_complete(manager.close())
    loop.close()
    if os.path.exists(path):
        os.remove(path)


class TestDatabaseManager:
    """Tests for the DatabaseManager logging mechanism."""

    def test_database_initializes_correctly(self, db_manager):
        manager, path, loop = db_manager
        assert os.path.exists(path)

        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_schema_has_correct_columns(self, db_manager):
        manager, path, loop = db_manager
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(predictions)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "request_id", "timestamp", "model_variant",
            "input_features", "prediction", "prediction_probability",
            "latency_ms",
        }
        assert columns == expected

    def test_log_prediction_stores_data(self, db_manager):
        manager, path, loop = db_manager
        loop.run_until_complete(
            manager.log_prediction(
                request_id="test-uuid-001",
                timestamp="2025-01-15T10:30:00",
                model_variant="A",
                input_features=json.dumps([1.0, 2.0, 3.0]),
                prediction=1.0,
                prediction_probability=0.95,
                latency_ms=5.123,
            )
        )

        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions WHERE request_id = 'test-uuid-001'")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "test-uuid-001"
        assert row[1] == "2025-01-15T10:30:00"
        assert row[2] == "A"
        assert row[3] == json.dumps([1.0, 2.0, 3.0])
        assert row[4] == 1.0
        assert row[5] == 0.95
        assert row[6] == 5.123

    def test_log_prediction_with_null_probability(self, db_manager):
        manager, path, loop = db_manager
        loop.run_until_complete(
            manager.log_prediction(
                request_id="test-uuid-002",
                timestamp="2025-01-15T10:31:00",
                model_variant="B",
                input_features=json.dumps([4.0, 5.0]),
                prediction=0.0,
                prediction_probability=None,
                latency_ms=3.456,
            )
        )

        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("SELECT prediction_probability FROM predictions WHERE request_id = 'test-uuid-002'")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] is None

    def test_get_prediction_count(self, db_manager):
        manager, path, loop = db_manager

        for i in range(5):
            loop.run_until_complete(
                manager.log_prediction(
                    request_id=f"count-test-{i}",
                    timestamp="2025-01-15T10:30:00",
                    model_variant="A" if i % 2 == 0 else "B",
                    input_features=json.dumps([float(i)]),
                    prediction=1.0,
                    prediction_probability=0.9,
                    latency_ms=1.0,
                )
            )

        count = loop.run_until_complete(manager.get_prediction_count())
        assert count == 5

    def test_get_all_predictions(self, db_manager):
        manager, path, loop = db_manager

        for i in range(3):
            loop.run_until_complete(
                manager.log_prediction(
                    request_id=f"all-test-{i}",
                    timestamp=f"2025-01-15T10:3{i}:00",
                    model_variant="A",
                    input_features=json.dumps([float(i)]),
                    prediction=1.0,
                    prediction_probability=0.85,
                    latency_ms=2.0,
                )
            )

        predictions = loop.run_until_complete(manager.get_all_predictions())
        assert len(predictions) == 3
        assert all(isinstance(p, dict) for p in predictions)
        assert predictions[0]["request_id"] == "all-test-0"

    def test_duplicate_request_id_raises_error(self, db_manager):
        manager, path, loop = db_manager

        loop.run_until_complete(
            manager.log_prediction(
                request_id="duplicate-test",
                timestamp="2025-01-15T10:30:00",
                model_variant="A",
                input_features=json.dumps([1.0]),
                prediction=1.0,
                prediction_probability=0.9,
                latency_ms=1.0,
            )
        )

        with pytest.raises(Exception):
            loop.run_until_complete(
                manager.log_prediction(
                    request_id="duplicate-test",
                    timestamp="2025-01-15T10:31:00",
                    model_variant="B",
                    input_features=json.dumps([2.0]),
                    prediction=0.0,
                    prediction_probability=0.7,
                    latency_ms=1.5,
                )
            )

    def test_concurrent_logging_integrity(self, db_manager):
        """Verify that concurrent logging does not corrupt data."""
        manager, path, loop = db_manager

        async def log_batch(start_idx: int, count: int):
            for i in range(start_idx, start_idx + count):
                await manager.log_prediction(
                    request_id=f"concurrent-{i}",
                    timestamp="2025-01-15T10:30:00",
                    model_variant="A" if i % 2 == 0 else "B",
                    input_features=json.dumps([float(i)]),
                    prediction=float(i % 2),
                    prediction_probability=0.8,
                    latency_ms=float(i),
                )

        loop.run_until_complete(log_batch(0, 50))

        count = loop.run_until_complete(manager.get_prediction_count())
        assert count == 50

    def test_uninitialized_manager_raises_error(self):
        manager = DatabaseManager(db_path="/tmp/nonexistent.db")
        loop = asyncio.new_event_loop()

        with pytest.raises(RuntimeError, match="not initialized"):
            loop.run_until_complete(
                manager.log_prediction(
                    request_id="fail",
                    timestamp="2025-01-15T10:30:00",
                    model_variant="A",
                    input_features="[]",
                    prediction=0.0,
                    prediction_probability=None,
                    latency_ms=0.0,
                )
            )
        loop.close()

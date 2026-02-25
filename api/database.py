"""Database management for A/B test prediction logging."""

import aiosqlite
import os
import logging

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.getenv("DATABASE_NAME", "data/ab_test_logs.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model_variant TEXT NOT NULL,
    input_features TEXT NOT NULL,
    prediction REAL NOT NULL,
    prediction_probability REAL,
    latency_ms REAL NOT NULL
)
"""

INSERT_PREDICTION_SQL = """
INSERT INTO predictions
    (request_id, timestamp, model_variant, input_features, prediction,
     prediction_probability, latency_ms)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


class DatabaseManager:
    """Async database manager for SQLite-based prediction logging.

    Uses aiosqlite for non-blocking database operations within the
    async FastAPI event loop.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self._connection: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create the database directory and predictions table if needed."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._connection = await aiosqlite.connect(self.db_path)
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA synchronous=NORMAL")
        await self._connection.execute(CREATE_TABLE_SQL)
        await self._connection.commit()
        logger.info("Database initialized at %s", self.db_path)

    async def close(self) -> None:
        """Close the database connection gracefully."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

    async def log_prediction(
        self,
        request_id: str,
        timestamp: str,
        model_variant: str,
        input_features: str,
        prediction: float,
        prediction_probability: float | None,
        latency_ms: float,
    ) -> None:
        """Log a prediction to the database.

        Args:
            request_id: Unique identifier for the request.
            timestamp: ISO-format timestamp of the request.
            model_variant: Which model variant handled this request ('A' or 'B').
            input_features: JSON-encoded input features string.
            prediction: The model's prediction value.
            prediction_probability: The prediction probability (if available).
            latency_ms: Request processing time in milliseconds.

        Raises:
            RuntimeError: If the database connection is not initialized.
        """
        if not self._connection:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        await self._connection.execute(
            INSERT_PREDICTION_SQL,
            (
                request_id,
                timestamp,
                model_variant,
                input_features,
                prediction,
                prediction_probability,
                latency_ms,
            ),
        )
        await self._connection.commit()

    async def get_all_predictions(self) -> list[dict]:
        """Retrieve all logged predictions.

        Returns:
            List of prediction records as dictionaries.

        Raises:
            RuntimeError: If the database connection is not initialized.
        """
        if not self._connection:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        self._connection.row_factory = aiosqlite.Row
        cursor = await self._connection.execute(
            "SELECT * FROM predictions ORDER BY timestamp"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_prediction_count(self) -> int:
        """Return total number of logged predictions.

        Raises:
            RuntimeError: If the database connection is not initialized.
        """
        if not self._connection:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        cursor = await self._connection.execute(
            "SELECT COUNT(*) FROM predictions"
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

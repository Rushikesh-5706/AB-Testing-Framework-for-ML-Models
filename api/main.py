"""FastAPI application for serving A/B tested ML model predictions.

This module implements the core REST API that loads two pre-trained ML model
variants and routes incoming prediction requests between them based on a
configurable traffic split ratio. Every prediction is logged to a persistent
SQLite database for downstream analysis.
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status

from api.database import DatabaseManager
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExperimentConfig,
    ExperimentConfigResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from api.traffic_splitter import TrafficSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_A_PATH = os.getenv("MODEL_A_PATH", "api/models/model_A.pkl")
MODEL_B_PATH = os.getenv("MODEL_B_PATH", "api/models/model_B.pkl")
DB_PATH = os.getenv("DATABASE_NAME", "data/ab_test_logs.db")
INITIAL_RATIO = float(os.getenv("MODEL_A_RATIO", "0.5"))

models: dict[str, object] = {}
splitter: TrafficSplitter = TrafficSplitter(model_a_ratio=INITIAL_RATIO)
db: DatabaseManager = DatabaseManager(db_path=DB_PATH)


def _load_model(path: str, label: str) -> object | None:
    """Load a serialized model from disk.

    Args:
        path: Filesystem path to the pickled model.
        label: Human-readable label for logging purposes.

    Returns:
        The deserialized model object, or None if the file is missing.
    """
    if not os.path.exists(path):
        logger.error("Model file not found: %s", path)
        return None
    model = joblib.load(path)
    logger.info("Loaded %s from %s", label, path)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown tasks."""
    models["A"] = _load_model(MODEL_A_PATH, "Model A")
    models["B"] = _load_model(MODEL_B_PATH, "Model B")

    loaded = [k for k, v in models.items() if v is not None]
    if not loaded:
        logger.critical("No models could be loaded. API will not serve predictions.")
    else:
        logger.info("Models loaded successfully: %s", loaded)

    await db.initialize()
    logger.info("Application startup complete")

    yield

    await db.close()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="A/B Testing Framework for ML Models",
    description=(
        "Production-grade REST API for serving and comparing ML model variants "
        "with traffic splitting, persistent logging, and statistical analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check the health status of the API, models, and database."""
    db_connected = False
    try:
        await db.get_prediction_count()
        db_connected = True
    except Exception:
        logger.exception("Database health check failed")

    return HealthResponse(
        status="healthy" if all(models.values()) and db_connected else "degraded",
        models_loaded={k: v is not None for k, v in models.items()},
        database_connected=db_connected,
        experiment_config=ExperimentConfigResponse(
            model_a_ratio=splitter.model_a_ratio,
            model_b_ratio=splitter.model_b_ratio,
            model_a_name="Logistic Regression",
            model_b_name="XGBoost",
        ),
    )


@app.get(
    "/experiment/config",
    response_model=ExperimentConfigResponse,
    tags=["Experiment"],
)
async def get_experiment_config() -> ExperimentConfigResponse:
    """Return the current experiment traffic split configuration."""
    return ExperimentConfigResponse(
        model_a_ratio=splitter.model_a_ratio,
        model_b_ratio=splitter.model_b_ratio,
        model_a_name="Logistic Regression",
        model_b_name="XGBoost",
    )


@app.post(
    "/experiment/config",
    response_model=ExperimentConfigResponse,
    tags=["Experiment"],
)
async def update_experiment_config(
    config: ExperimentConfig,
) -> ExperimentConfigResponse:
    """Update the traffic split ratio for the experiment.

    Args:
        config: New experiment configuration with model_a_ratio.

    Returns:
        Updated experiment configuration.
    """
    try:
        splitter.update_ratio(config.model_a_ratio)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    logger.info(
        "Traffic split updated: Model A=%.1f%%, Model B=%.1f%%",
        splitter.model_a_ratio * 100,
        splitter.model_b_ratio * 100,
    )
    return ExperimentConfigResponse(
        model_a_ratio=splitter.model_a_ratio,
        model_b_ratio=splitter.model_b_ratio,
        model_a_name="Logistic Regression",
        model_b_name="XGBoost",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Route a prediction request to a model variant and log the result.

    The traffic splitter assigns the request to Model A or Model B based on
    the current split ratio. If a session_id is provided, the assignment is
    deterministic (sticky).

    Args:
        request: Prediction request containing feature values.

    Returns:
        Prediction response with the result, variant, and latency.

    Raises:
        HTTPException: If no models are available or prediction fails.
    """
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    variant = splitter.assign_variant(session_id=request.session_id)
    model = models.get(variant)

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model variant {variant} is not available",
        )

    try:
        features_array = np.array(request.features).reshape(1, -1)
        prediction = float(model.predict(features_array)[0])

        prediction_probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_array)[0]
            prediction_probability = float(max(probabilities))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc

    latency_ms = (time.perf_counter() - start_time) * 1000
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        await db.log_prediction(
            request_id=request_id,
            timestamp=timestamp,
            model_variant=variant,
            input_features=json.dumps(request.features),
            prediction=prediction,
            prediction_probability=prediction_probability,
            latency_ms=round(latency_ms, 3),
        )
    except Exception:
        logger.exception("Failed to log prediction %s", request_id)

    return PredictionResponse(
        request_id=request_id,
        model_variant=variant,
        prediction=prediction,
        prediction_probability=prediction_probability,
        latency_ms=round(latency_ms, 3),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
)
async def predict_batch(
    batch_request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Process a batch of prediction requests.

    Each request in the batch is independently assigned a model variant
    and logged to the database.

    Args:
        batch_request: Batch of prediction requests.

    Returns:
        Batch response with all individual prediction results.
    """
    batch_start = time.perf_counter()
    predictions = []

    for single_request in batch_request.requests:
        result = await predict(single_request)
        predictions.append(result)

    total_latency_ms = (time.perf_counter() - batch_start) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        total_latency_ms=round(total_latency_ms, 3),
    )

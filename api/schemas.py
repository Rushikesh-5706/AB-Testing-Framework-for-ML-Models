"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """Schema for a single prediction request."""

    features: list[float] = Field(
        ...,
        description="List of numerical feature values for the ML model",
        min_length=1,
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for sticky traffic assignment",
    )


class PredictionResponse(BaseModel):
    """Schema for a single prediction response."""

    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    model_variant: str
    prediction: float
    prediction_probability: Optional[float] = None
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""

    requests: list[PredictionRequest] = Field(
        ...,
        description="List of prediction requests",
        min_length=1,
    )


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses."""

    predictions: list[PredictionResponse]
    total_latency_ms: float


class ExperimentConfig(BaseModel):
    """Schema for experiment configuration."""

    model_config = ConfigDict(protected_namespaces=())

    model_a_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ratio of traffic routed to Model A (0.0 to 1.0)",
    )


class ExperimentConfigResponse(BaseModel):
    """Schema for experiment configuration response."""

    model_config = ConfigDict(protected_namespaces=())

    model_a_ratio: float
    model_b_ratio: float
    model_a_name: str
    model_b_name: str


class HealthResponse(BaseModel):
    """Schema for health check response."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    models_loaded: dict[str, bool]
    database_connected: bool
    experiment_config: ExperimentConfigResponse

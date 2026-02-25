# Production A/B Testing Framework for ML Models

A production-grade A/B testing framework for comparing machine learning model variants with statistical rigor. The system handles experimental design, traffic splitting, metric tracking, and automated statistical analysis through a containerized REST API, persistent database, and interactive dashboard.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Network                    │
│                                                             │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │   FastAPI Service     │    │   Streamlit Dashboard     │  │
│  │   (Port 8000)         │    │   (Port 8501)             │  │
│  │                       │    │                           │  │
│  │  ┌─────────────────┐ │    │  ┌──────────────────────┐ │  │
│  │  │ Traffic Splitter │ │    │  │ Metric Visualizations│ │  │
│  │  │ (Configurable)   │ │    │  │ Statistical Tests    │ │  │
│  │  └────────┬────────┘ │    │  │ Recommendation       │ │  │
│  │           │           │    │  └──────────┬───────────┘ │  │
│  │  ┌────────▼────────┐ │    │             │              │  │
│  │  │  Model A (LR)   │ │    │  ┌──────────▼───────────┐ │  │
│  │  │  Model B (XGB)  │ │    │  │ Analysis Pipeline    │ │  │
│  │  └────────┬────────┘ │    │  │ (run_analysis.py)    │ │  │
│  │           │           │    │  └──────────┬───────────┘ │  │
│  │  ┌────────▼────────┐ │    │             │              │  │
│  │  │ Async SQLite DB │◄├────├─────────────┘              │  │
│  │  │ (WAL Mode)      │ │    │                           │  │
│  │  └─────────────────┘ │    │                           │  │
│  └──────────────────────┘    └──────────────────────────┘  │
│                                                             │
│  Shared Volume: ./data/ab_test_logs.db                      │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
.
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI application with prediction endpoints
│   ├── database.py          # Async SQLite database manager
│   ├── traffic_splitter.py  # Configurable A/B traffic routing logic
│   ├── schemas.py           # Pydantic request/response models
│   ├── requirements.txt     # Python dependencies (pinned versions)
│   └── models/
│       ├── model_A.pkl      # Logistic Regression (serialized pipeline)
│       ├── model_B.pkl      # XGBoost classifier (serialized)
│       ├── model_metadata.json  # Training metrics and feature info
│       └── sample_input.json    # Sample feature vector for testing
├── analysis/
│   ├── __init__.py
│   ├── run_analysis.py      # Statistical analysis pipeline
│   └── dashboard.py         # Streamlit interactive dashboard
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared fixtures and mock models
│   ├── test_api.py          # API endpoint tests
│   ├── test_traffic_splitter.py  # Traffic splitting distribution tests
│   └── test_logging.py      # Database logging mechanism tests
├── scripts/
│   └── simulate_traffic.py  # Traffic simulation tool
├── data/                    # Persistent database storage (volume-mounted)
├── train_models.py          # Model training script
├── Dockerfile               # Multi-stage production Docker image
├── .dockerignore            # Docker build exclusions
├── docker-compose.yml       # Service orchestration
├── submission.yml           # Automated build/deploy/test/analyze commands
├── METHODOLOGY.md           # Experimental design documentation
└── README.md                # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- pip

### 1. Install Dependencies

```bash
pip install -r api/requirements.txt
```

### 2. Train Models

```bash
python train_models.py
```

This trains two model variants on the Breast Cancer Wisconsin dataset:
- **Model A**: Logistic Regression with StandardScaler preprocessing
- **Model B**: XGBoost gradient-boosted classifier

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Start with Docker Compose

```bash
docker-compose build
docker-compose up -d
```

### 5. Simulate Traffic

```bash
python scripts/simulate_traffic.py --num-requests 500
```

### 6. Run Analysis

```bash
python analysis/run_analysis.py
```

### 7. View Dashboard

Open [http://localhost:8501](http://localhost:8501) in your browser.

## API Reference

### `POST /predict`

Route a prediction request to a model variant.

**Request:**
```json
{
  "features": [17.99, 10.38, 122.8, ...],
  "session_id": "optional-user-id"
}
```

**Response:**
```json
{
  "request_id": "uuid",
  "model_variant": "A",
  "prediction": 1.0,
  "prediction_probability": 0.95,
  "latency_ms": 3.45
}
```

### `POST /predict/batch`

Process multiple prediction requests in one call.

**Request:**
```json
{
  "requests": [
    {"features": [17.99, 10.38, ...]},
    {"features": [13.54, 14.36, ...]}
  ]
}
```

### `GET /health`

Returns system health including model status and database connectivity.

### `GET /experiment/config`

Returns the current traffic split configuration.

### `POST /experiment/config`

Update the traffic split ratio at runtime.

```json
{
  "model_a_ratio": 0.7
}
```

## Traffic Splitting

The framework supports two routing modes:

1. **Random Assignment**: Each request is independently routed based on the configured probability (default 50/50).
2. **Session-Based Sticky Assignment**: When a `session_id` is provided, the same user always sees the same model variant, achieved via deterministic SHA-256 hashing.

The split ratio is configurable at runtime via the `/experiment/config` endpoint.

## Analysis Pipeline

The analysis pipeline (`analysis/run_analysis.py`) computes:

**Performance Metrics (per variant):**
- Request count and traffic share
- Mean, median, P95, and P99 latency
- Mean prediction and positive prediction rate
- Prediction probability statistics

**Statistical Tests:**
- **Welch's t-test** — continuous metric comparison (latency)
- **Mann-Whitney U test** — non-parametric prediction distribution comparison
- **Chi-squared test** — categorical association between variant and prediction class

Results are saved to `analysis/results.json` and visualized in the Streamlit dashboard.

## Testing

The test suite covers three critical areas:

| Test File | Coverage Area | Number of Tests |
|-----------|--------------|-----------------|
| `test_api.py` | API endpoints, validation, logging, session stickiness | 30 |
| `test_traffic_splitter.py` | Distribution accuracy, stickiness, edge cases | 16 |
| `test_logging.py` | Schema, persistence, concurrency, error handling | 9 |
| `test_analysis.py` | Metrics, statistical tests, recommendations | 23 |
| **Total** | **Full system coverage** | **82** |

Run the full suite:

```bash
pytest tests/ -v --tb=short
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_NAME` | `data/ab_test_logs.db` | SQLite database path |
| `MODEL_A_PATH` | `api/models/model_A.pkl` | Path to Model A |
| `MODEL_B_PATH` | `api/models/model_B.pkl` | Path to Model B |
| `MODEL_A_RATIO` | `0.5` | Initial traffic ratio for Model A |
| `API_URL` | `http://localhost:8000` | API URL for simulation script |
| `RESULTS_PATH` | `analysis/results.json` | Analysis output path |

## Docker

### Build

```bash
docker-compose build
```

### Run

```bash
docker-compose up -d
```

### Logs

```bash
docker-compose logs -f api
docker-compose logs -f dashboard
```

### Stop

```bash
docker-compose down
```

## Submission Commands

As defined in `submission.yml`:

```bash
# Setup: install deps, train models, build images
# Deploy: start services, verify health
# Test: run pytest suite
# Analyze: simulate traffic, run analysis, view dashboard
```

# Production A/B Testing Framework for ML Models

A production-grade A/B testing framework designed for comparing machine learning model variants with statistical rigor. This system handles experimental design, configurable traffic splitting, persistent metric tracking, and automated statistical analysis through a containerized REST API, SQLite database, and interactive Streamlit dashboard.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Setup and Installation](#setup-and-installation)
5. [Running the Application](#running-the-application)
6. [API Reference](#api-reference)
7. [Traffic Splitting Logic](#traffic-splitting-logic)
8. [Analysis Pipeline](#analysis-pipeline)
9. [Dashboard](#dashboard)
10. [Testing](#testing)
11. [Docker Deployment](#docker-deployment)
12. [Environment Variables](#environment-variables)
13. [Submission Commands](#submission-commands)

---

## Architecture Overview

The framework consists of three core services working together through a shared SQLite database:

```text
                          Incoming Prediction Requests
                                     |
                                     v
                    +--------------------------------+
                    |        FastAPI Service         |
                    |        (Port 8000)             |
                    |                                |
                    |   +------------------------+   |
                    |   |    Traffic Splitter    |   |
                    |   | (configurable 50/50,   |   |
                    |   |  70/30, session-sticky)|   |
                    |   +-------+--------+-------+   |
                    |           |        |           |
                    |           v        v           |
                    |   +-------+---+ +--+-------+   |
                    |   | Model A   | | Model B  |   |
                    |   | Logistic  | | XGBoost  |   |
                    |   | Regr.     | | Class.   |   |
                    |   +-----------+ +----------+   |
                    |           |        |           |
                    |           v        v           |
                    |   +------------------------+   |
                    |   |  Async SQLite Logger   |   |
                    |   |  (WAL mode via         |   |
                    |   |   aiosqlite)           |   |
                    |   +-----------+------------+   |
                    +---------------|----------------+
                                    |
                                    v
                    +--------------------------------+
                    |    SQLite Database             |
                    |    data/ab_test_logs.db        |
                    |    (persistent volume mount)   |
                    +---------------|----------------+
                                    |
                    +---------------+----------------+
                    |                                |
                    v                                v
    +---------------------------+   +---------------------------+
    |   Analysis Pipeline       |   |   Streamlit Dashboard     |
    |   (run_analysis.py)       |   |   (Port 8501)             |
    |                           |   |                           |
    |   - Welch's t-test        |   |   - KPI Summary           |
    |   - Mann-Whitney U        |   |   - Traffic Distribution  |
    |   - Chi-squared test      |   |   - Latency Box Plots     |
    |   - Recommendation        |   |   - Prediction Histograms |
    +---------------------------+   |   - Significance Tests    |
                                    +---------------------------+
```

### Data Flow

1. A prediction request arrives at the FastAPI service.
2. The traffic splitter assigns the request to Model A or Model B based on the configured ratio.
3. The assigned model generates a prediction.
4. All request metadata (request ID, timestamp, variant, features, prediction, probability, latency) is logged to the SQLite database asynchronously.
5. The analysis pipeline reads from the database, computes per-variant metrics, and runs statistical significance tests.
6. The Streamlit dashboard visualizes the results and displays a recommendation.

---

## Project Structure

```
.
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application with prediction endpoints
│   ├── database.py              # Async SQLite database manager (aiosqlite)
│   ├── traffic_splitter.py      # Configurable A/B traffic routing logic
│   ├── schemas.py               # Pydantic request/response models
│   ├── requirements.txt         # Python dependencies (pinned versions)
│   └── models/
│       ├── model_A.pkl          # Logistic Regression pipeline (serialized)
│       ├── model_B.pkl          # XGBoost classifier (serialized)
│       ├── model_metadata.json  # Training metrics and feature information
│       └── sample_input.json    # Sample feature vector for testing
├── analysis/
│   ├── __init__.py
│   ├── run_analysis.py          # Statistical analysis pipeline
│   └── dashboard.py             # Streamlit interactive dashboard
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared test fixtures and mock models
│   ├── test_api.py              # API endpoint tests (30 tests)
│   ├── test_traffic_splitter.py # Traffic splitting tests (16 tests)
│   ├── test_logging.py          # Database logging tests (9 tests)
│   └── test_analysis.py         # Analysis pipeline tests (23 tests)
├── scripts/
│   └── simulate_traffic.py      # Traffic simulation tool
├── data/                        # Persistent database storage (volume-mounted)
├── train_models.py              # Model training and serialization script
├── Dockerfile                   # Multi-stage production Docker image
├── .dockerignore                # Docker build exclusions
├── docker-compose.yml           # Service orchestration (API + Dashboard)
├── pytest.ini                   # Pytest configuration with markers
├── submission.yml               # Automated setup/deploy/test/analyze
├── METHODOLOGY.md               # Experimental design documentation
└── README.md
```

---

## Prerequisites

Before running this project, ensure the following are installed:

| Dependency | Minimum Version | Purpose |
|------------|----------------|---------|
| Python | 3.11+ | Runtime |
| pip | 21.0+ | Package management |
| Docker | 20.0+ | Containerization |
| Docker Compose | 2.0+ | Service orchestration |

---

## Setup and Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Rushikesh-5706/AB-Testing-Framework-for-ML-Models.git
cd AB-Testing-Framework-for-ML-Models
```

### Step 2: Install Python Dependencies

```bash
pip install -r api/requirements.txt
```

### Step 3: Train the ML Models

This trains two model variants on the Breast Cancer Wisconsin dataset (built into scikit-learn, no external downloads required):

```bash
python3 train_models.py
```

Expected output:

```
Training Model A (Logistic Regression)...
  Model A Metrics: {'accuracy': 0.9825, 'f1_score': 0.9861, 'roc_auc': 0.9954}
  Saved to api/models/model_A.pkl

Training Model B (XGBoost)...
  Model B Metrics: {'accuracy': 0.9474, 'f1_score': 0.9589, 'roc_auc': 0.9924}
  Saved to api/models/model_B.pkl

Model training complete.
  Model A (98.2% accuracy) vs Model B (94.7% accuracy)
```

| Model | Algorithm | Accuracy | F1 Score | ROC AUC |
|-------|-----------|----------|----------|---------|
| Model A | Logistic Regression (with StandardScaler) | 98.25% | 0.9861 | 0.9954 |
| Model B | XGBoost Classifier | 94.74% | 0.9589 | 0.9924 |

### Step 4: Run the Automated Tests

```bash
pytest tests/ -v
```

All 82 tests should pass:

```
tests/test_analysis.py      - 23 passed
tests/test_api.py            - 30 passed
tests/test_logging.py        -  9 passed
tests/test_traffic_splitter.py - 16 passed
========================= 82 passed in 1.44s =========================
```

---

## Running the Application

### Option A: Run Locally (without Docker)

Start the FastAPI server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In a separate terminal, simulate traffic:

```bash
python3 scripts/simulate_traffic.py --num-requests 500
```

Run the analysis pipeline:

```bash
python3 analysis/run_analysis.py
```

Start the dashboard:

```bash
streamlit run analysis/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### Option B: Run with Docker Compose

```bash
docker-compose build
docker-compose up -d
```

This starts two services:
- API on port 8000
- Dashboard on port 8501

Verify the API is healthy:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "models_loaded": {"A": true, "B": true},
  "database_connected": true,
  "experiment_config": {
    "model_a_ratio": 0.5,
    "model_b_ratio": 0.5,
    "model_a_name": "Logistic Regression",
    "model_b_name": "XGBoost"
  }
}
```

Simulate traffic:

```bash
python3 scripts/simulate_traffic.py --num-requests 500
```

Run analysis:

```bash
python3 analysis/run_analysis.py
```

View the dashboard at: http://localhost:8501

Stop services:

```bash
docker-compose down
```

---

## API Reference

The API exposes five endpoints:

### POST /predict

Route a prediction request to a model variant.

Request body:

```json
{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
  "session_id": "user-42"
}
```

Response:

```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "model_variant": "A",
  "prediction": 1.0,
  "prediction_probability": 0.9543,
  "latency_ms": 1.234
}
```

The `session_id` field is optional. When provided, the same user is always routed to the same model variant (sticky assignment via SHA-256 hashing).

### POST /predict/batch

Process multiple prediction requests in a single call.

Request body:

```json
{
  "requests": [
    {"features": [17.99, 10.38, ...]},
    {"features": [13.54, 14.36, ...]}
  ]
}
```

### GET /health

Returns system health including model status and database connectivity.

### GET /experiment/config

Returns the current traffic split configuration.

### POST /experiment/config

Update the traffic split ratio at runtime without redeployment.

```json
{
  "model_a_ratio": 0.7
}
```

---

## Traffic Splitting Logic

The framework supports two routing modes:

| Mode | Behavior | Use Case |
|------|----------|----------|
| Random | Each request is independently assigned based on probability | Stateless testing |
| Session-Sticky | Same session ID always routes to same variant (SHA-256 hash) | User-consistent testing |

The split ratio is configurable at runtime via the `/experiment/config` POST endpoint. The default is a 50/50 split.

The traffic splitter is thread-safe with `threading.RLock` and supports ratio updates without service restart.

---

## Analysis Pipeline

The analysis script (`analysis/run_analysis.py`) connects to the SQLite database and computes:

### Per-Variant Metrics

| Metric | Description |
|--------|-------------|
| Request count | Number of requests handled by each variant |
| Traffic share | Percentage of total traffic per variant |
| Mean prediction | Average prediction value |
| Positive prediction rate | Proportion of positive class predictions |
| Mean latency | Average processing time in milliseconds |
| P95 latency | 95th percentile latency |
| P99 latency | 99th percentile latency |
| Mean probability | Average prediction confidence score |

### Statistical Significance Tests

| Test | Metric Type | Purpose |
|------|-------------|---------|
| Welch's t-test | Continuous (latency) | Compare mean latency between variants |
| Mann-Whitney U | Non-parametric (prediction) | Compare prediction distributions |
| Chi-squared | Categorical (prediction class) | Test association between variant and outcome |
| Welch's t-test | Continuous (probability) | Compare mean confidence scores |

All tests use a significance level of alpha = 0.05.

Run the analysis:

```bash
python3 analysis/run_analysis.py
```

Results are saved to `analysis/results.json`.

---

## Dashboard

The Streamlit dashboard (`analysis/dashboard.py`) provides interactive visualizations:

- Summary KPIs for each model variant (latency, positive rate)
- Traffic distribution pie chart
- Latency comparison box plots and histograms
- Prediction class distribution
- Prediction probability distribution
- Statistical significance indicators with p-values
- Time-series of request volume
- Detailed metrics table
- Raw prediction log explorer
- Automated recommendation based on analysis results

Start the dashboard:

```bash
streamlit run analysis/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

Access at: http://localhost:8501

---

## Testing

The test suite consists of 82 automated tests organized into four modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_api.py` | 30 | All API endpoints, request validation, response structure, database logging, session stickiness |
| `test_traffic_splitter.py` | 16 | Distribution accuracy at 50/50, 70/30, 90/10 ratios, boundary cases, sticky assignment, runtime updates |
| `test_logging.py` | 9 | Database schema, data persistence, null handling, concurrent writes, duplicate rejection, uninitialized access |
| `test_analysis.py` | 23 | Data loading, metrics computation, statistical tests, p-value ranges, recommendation logic |
| **Total** | **82** | |

### Running Tests

Run all tests:

```bash
pytest tests/ -v
```

Run only unit tests:

```bash
pytest tests/ -v -m unit
```

Run only integration tests:

```bash
pytest tests/ -v -m integration
```

---

## Docker Deployment

### Build the Image

```bash
docker-compose build
```

The Dockerfile uses a multi-stage build:
- Builder stage installs dependencies
- Production stage copies only runtime artifacts
- Runs as a non-root user (`appuser`)
- Includes a health check

### Start Services

```bash
docker-compose up -d
```

### View Logs

```bash
docker-compose logs -f api
docker-compose logs -f dashboard
```

### Stop Services

```bash
docker-compose down
```

### Docker Hub

The image is available on Docker Hub:

```bash
docker pull rushi5706/ab-testing-framework:latest
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_NAME` | `data/ab_test_logs.db` | Path to the SQLite database file |
| `MODEL_A_PATH` | `api/models/model_A.pkl` | Path to serialized Model A |
| `MODEL_B_PATH` | `api/models/model_B.pkl` | Path to serialized Model B |
| `MODEL_A_RATIO` | `0.5` | Initial traffic ratio for Model A (0.0 to 1.0) |
| `API_URL` | `http://localhost:8000` | API URL used by the simulation script |
| `RESULTS_PATH` | `analysis/results.json` | Path where analysis results are saved |

---

## Submission Commands

As defined in `submission.yml`:

### Setup

```bash
pip install -r api/requirements.txt
python3 train_models.py
docker-compose build
```

### Deploy

```bash
docker-compose up -d
sleep 10
curl -s http://localhost:8000/health | python3 -m json.tool
```

### Test

```bash
pytest tests/ -v --tb=short
```

### Analyze

```bash
python3 scripts/simulate_traffic.py --num-requests 500 --delay-ms 5
python3 analysis/run_analysis.py
```

The dashboard is accessible at http://localhost:8501 after running `docker-compose up -d`.

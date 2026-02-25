FROM python:3.11-slim AS builder

WORKDIR /build

COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r api/requirements.txt

FROM python:3.11-slim

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /install /usr/local

COPY api/ ./api/
COPY analysis/ ./analysis/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY train_models.py ./
COPY pytest.ini ./

RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

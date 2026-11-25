FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".[api,dashboard]"

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=app:app artifacts/gbm.joblib ./model/gbm.joblib
COPY --chown=app:app dashboard ./dashboard
COPY --chown=app:app .streamlit ./.streamlit
RUN mkdir -p /app/data && chown app:app /app/data

USER app
ENV MODEL_PATH=/app/model/gbm.joblib \
    POLICY_PATH=/app/data/policy.json \
    REVIEW_LOG_PATH=/app/data/review_log.csv \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "credit_underwriting.api:app", "--host", "0.0.0.0", "--port", "8000"]

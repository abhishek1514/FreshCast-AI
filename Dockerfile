# Multi-stage production container for FreshCast Demand Forecasting System
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system compilation dependencies & OpenMP for LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for deterministic dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency specifications
COPY pyproject.toml .python-version ./

# Create virtualenv and install dependencies
RUN uv venv /app/.venv --python 3.11
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source tree and install package
COPY src ./src
COPY README.md ./
RUN uv pip install --no-cache -e .

# Final runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime OpenMP library for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv and application artifacts from builder
COPY --from=builder /app/.venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy code and configs
COPY configs ./configs
COPY src ./src
COPY app ./app
COPY data ./data
COPY models ./models
COPY dvc.yaml ./
COPY README.md ./

# Create non-root application user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8501 8000 5000

# Default command: launch the FastAPI REST service (override in compose or cli)
CMD ["uvicorn", "freshcast.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

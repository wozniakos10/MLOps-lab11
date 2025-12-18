# Dockerfile.dev
# Build stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
# Sync only inference group to .venv
RUN uv sync --frozen --group inference --no-install-project

# Runtime stage
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy virtual environment
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Add src to PYTHONPATH to now have errors with import modules
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Copy application code
COPY src ./src

# Copy model artifacts (ONNX + tokenizer.json + classifier.joblib)
COPY models ./models

# Expose the application port
EXPOSE 8000



ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["src.lab11_lib.app.handler"]
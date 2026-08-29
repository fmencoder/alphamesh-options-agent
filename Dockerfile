# AlphaMesh runtime image.
#
# No secret is ever baked in. Every credential arrives at run time through the
# environment (Railway variables, docker --env-file, etc.).

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ALPACA_PAPER=true

WORKDIR /app

COPY pyproject.toml README.md ./
COPY alphamesh ./alphamesh
RUN pip install --no-cache-dir ".[dashboard]"

COPY config ./config
COPY dashboard ./dashboard
COPY data/mcp_capture ./data/mcp_capture

# The journal lives on a mounted volume in production so it survives redeploys.
RUN mkdir -p /app/data && useradd --create-home --uid 10001 alphamesh \
    && chown -R alphamesh:alphamesh /app
USER alphamesh

ENV DATABASE_PATH=/app/data/alphamesh.db

# Fails non-zero when the paper guard does not pass, so an unsafe container
# never reports healthy.
HEALTHCHECK --interval=60s --timeout=15s --start-period=20s --retries=3 \
    CMD python -m alphamesh.main health || exit 1

# Default to the agent. Override with the dashboard command for the web service.
CMD ["python", "-m", "alphamesh.main", "run"]

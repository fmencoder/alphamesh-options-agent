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
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# The audit journal lives on a mounted volume so it survives redeploys and
# restarts. /data is created and owned here so the image also runs standalone
# with no volume attached; the entrypoint verifies writability either way.
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && useradd --create-home --uid 10001 alphamesh \
    && mkdir -p /data \
    && chown -R alphamesh:alphamesh /app /data

USER alphamesh

ENV DATABASE_PATH=/data/alphamesh.db \
    ALPHAMESH_LOOP_SECONDS=60 \
    ALPHAMESH_CLOSED_POLL_SECONDS=300

# Fails non-zero when the paper guard does not pass or an execution-critical
# dependency is unreachable, so an unsafe container never reports healthy.
HEALTHCHECK --interval=60s --timeout=20s --start-period=30s --retries=3 \
    CMD python -m alphamesh.main health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default to the agent. Override with the dashboard command for the web service.
CMD ["python", "-m", "alphamesh.main", "run"]

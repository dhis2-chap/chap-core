# THIS DOCKERFILE RUNS THE WEB API

# Use a slim base image with Python and uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Use copy mode for mounted volumes
ENV UV_LINK_MODE=copy

# Create non-root user early
RUN useradd -m -s /bin/bash chap && \
    mkdir -p /app && chown -R chap:chap /app

# Set working directory
WORKDIR /app

# Install required system packages efficiently
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends git tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files and assign ownership to 'chap'
COPY ./pyproject.toml .
COPY ./uv.lock .
COPY ./.python-version .python-version
COPY ./chap_core ./chap_core
COPY ./config ./config
COPY ./scripts/seed.py ./scripts/seed.py
COPY ./README.md .

# Install only production dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Switch to non-root user
USER chap
EXPOSE 8000

ENV PORT=8000
ENV TIMEOUT=60
ENV GRACEFUL_TIMEOUT=30
ENV KEEPALIVE=5
ENV FORWARDED_ALLOW_IPS="*"

# Ensure virtual environment is first in PATH
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["sh","-c", "\
    : ${FORWARDED_ALLOW_IPS:='*'}; \
    if [ -z \"$WORKERS\" ]; then \
    WORKERS=$(( ( $(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 2) * 2 ) + 1 )); \
    fi; \
    exec gunicorn -k uvicorn.workers.UvicornWorker chap_core.rest_api.v1.rest_api:app \
    --bind 0.0.0.0:${PORT} \
    --workers ${WORKERS} \
    --timeout ${TIMEOUT} \
    --graceful-timeout ${GRACEFUL_TIMEOUT} \
    --keep-alive ${KEEPALIVE} \
    --forwarded-allow-ips=${FORWARDED_ALLOW_IPS} \
    --max-requests 1000 \
    --max-requests-jitter 200 \
    --access-logfile - \
    --error-logfile - \
    --worker-tmp-dir /dev/shm \
    "]

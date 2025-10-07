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

# Set the default port Gunicorn will bind to inside the container.
# Can be overridden at runtime with: `docker run -e PORT=9000 -p 9000:9000 myimage`
ENV PORT=8000

# Worker timeout (in seconds).  
# If a worker does not respond within this time, Gunicorn will kill it.  
# Helps prevent "hung" workers from stalling the app.
ENV TIMEOUT=60

# Graceful timeout (in seconds).  
# How long Gunicorn will wait for workers to finish ongoing requests  
# before forcefully killing them during a restart or reload.
ENV GRACEFUL_TIMEOUT=30

# Keep-alive (in seconds).  
# The amount of time to wait for the next request on a persistent HTTP connection  
# before closing it.
ENV KEEPALIVE=5

# List of IPs (or `*`) from which Gunicorn will trust X-Forwarded-* headers.  
# Required when running behind a reverse proxy (like Nginx, Traefik, or Docker ingress).  
# `*` means "trust all", which is fine in container setups where the proxy is controlled.
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

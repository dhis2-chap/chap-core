FROM ghcr.io/astral-sh/uv:0.8-python3.13-bookworm-slim

# OCI labels for container metadata
LABEL org.opencontainers.image.title="Chap Modeling Platform"
LABEL org.opencontainers.image.description="The backend engine for the Chap Modeling Platform"
LABEL org.opencontainers.image.vendor="DHIS2"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"
LABEL org.opencontainers.image.source="https://github.com/dhis2-chap/chap-core"

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

RUN useradd -m -s /bin/bash chap && \
    mkdir -p /app && chown -R chap:chap /app

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends git tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --chown=chap:chap ./pyproject.toml ./uv.lock ./.python-version ./README.md ./
COPY --chown=chap:chap ./chap_core ./chap_core
COPY --chown=chap:chap ./config ./config
COPY --chown=chap:chap ./gunicorn.conf.py ./gunicorn.conf.py
COPY --chown=chap:chap ./alembic.ini ./alembic.ini
COPY --chown=chap:chap ./alembic ./alembic

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

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

# Max requests per worker before restart (helps prevent memory leaks).
# Can be overridden at runtime with: `docker run -e MAX_REQUESTS=2000 myimage`
ENV MAX_REQUESTS=1000

# Max requests jitter adds randomness to prevent all workers restarting simultaneously.
ENV MAX_REQUESTS_JITTER=200

# Ensure virtual environment is first in PATH
ENV PATH="/app/.venv/bin:$PATH"

# Health check to verify the API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health').read()" || exit 1

ENTRYPOINT ["/usr/bin/tini","--"]

CMD ["sh","-c", "\
effective_cpus() { \
  base=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1); \
  if read -r quota period < /sys/fs/cgroup/cpu.max 2>/dev/null; then \
    if [ \"$quota\" != \"max\" ]; then \
      echo $(( (quota + period - 1) / period )); return; \
    fi; \
  fi; \
  echo \"$base\"; \
}; \
: ${FORWARDED_ALLOW_IPS:='*'}; \
CPUS=$(effective_cpus); \
: ${WORKERS:=$(( CPUS * 2 + 1 ))}; \
exec gunicorn -c gunicorn.conf.py -k uvicorn.workers.UvicornWorker chap_core.rest_api.v1.rest_api:app \
  --bind 0.0.0.0:${PORT} \
  #--workers ${WORKERS} \
  --workers 1 \
  --timeout ${TIMEOUT} \
  --graceful-timeout ${GRACEFUL_TIMEOUT} \
  --keep-alive ${KEEPALIVE} \
  --forwarded-allow-ips=${FORWARDED_ALLOW_IPS} \
  --max-requests ${MAX_REQUESTS} \
  --max-requests-jitter ${MAX_REQUESTS_JITTER} \
  --access-logfile - \
  --error-logfile - \
  --worker-tmp-dir /dev/shm \
"]

FROM ghcr.io/astral-sh/uv:0.9-python3.13-bookworm-slim

# Compile Python bytecode for faster startup (.venv only)
ENV UV_COMPILE_BYTECODE=1

# Use copy mode to avoid issues with file permissions in mixed environments
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt upgrade -y && \
    apt install -y --no-install-recommends git curl && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
\
    useradd --create-home --shell /usr/sbin/nologin chap

WORKDIR /app

# TODO: Figure out what's writing to the folder and make it write elsewhere
RUN chown -R chap:chap /app

COPY --chown=root:root ./pyproject.toml ./uv.lock ./.python-version ./gunicorn.conf.py ./alembic.ini README.md ./
COPY --chown=root:root ./chap_core ./chap_core
COPY --chown=root:root ./config ./config
COPY --chown=root:root ./alembic ./alembic

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PORT=8000

ENV PATH="/app/.venv/bin:$PATH"

HEALTHCHECK CMD curl --fail http://localhost:${PORT}/health || exit 1

# Pre-compile application bytecode as root user
RUN python -m compileall -q chap_core/

# Prevent Python from writing .pyc files at runtime
ENV PYTHONDONTWRITEBYTECODE=1

USER chap

CMD [ \
    "gunicorn", "-c", "gunicorn.conf.py", "-k", "uvicorn.workers.UvicornWorker", "chap_core.rest_api.v1.rest_api:app", \
      "--bind 0.0.0.0:${PORT}", \
      "--workers 1", \
      "--timeout 60", \
      "--graceful-timeout 30", \
      "--keep-alive 5", \
      "--forwarded-allow-ips *", \
      "--max-requests 1000", \
      "--max-requests-jitter 200", \
      "--access-logfile -", \
      "--error-logfile -", \
      "--worker-tmp-dir /dev/shm" \
  ]

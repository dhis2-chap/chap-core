FROM ghcr.io/astral-sh/uv:0.10-python3.13-trixie-slim

# Compile Python bytecode for faster startup (.venv only)
ENV UV_COMPILE_BYTECODE=1
# Use copy mode to avoid issues with file permissions in mixed environments
ENV UV_LINK_MODE=copy
# Prevent Python from writing .pyc files at runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLCONFIGDIR=/tmp
ENV PORT=8000
ENV PATH="/app/.venv/bin:$PATH"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt upgrade -y && \
    apt install -y --no-install-recommends git curl && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
\
    useradd --no-create-home --shell /usr/sbin/nologin chap

WORKDIR /app

COPY --chown=root:root ./pyproject.toml ./uv.lock ./.python-version ./gunicorn.conf.py ./alembic.ini README.md ./
COPY --chown=root:root ./chap_core ./chap_core
COPY --chown=root:root ./config ./config
COPY --chown=root:root ./alembic ./alembic

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev && \
    python -m compileall -q chap_core/

HEALTHCHECK CMD curl --fail http://localhost:${PORT}/health || exit 1

USER chap

CMD [ \
    "gunicorn", "-c", "gunicorn.conf.py", "-k", "uvicorn.workers.UvicornWorker", "chap_core.rest_api.app:app", \
      "--bind 0.0.0.0:${PORT}", \
      "--workers 1" \
  ]

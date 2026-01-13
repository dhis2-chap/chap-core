FROM ghcr.io/astral-sh/uv:0.8-python3.13-bookworm-slim

LABEL org.opencontainers.image.title="Chap Modeling Platform"
LABEL org.opencontainers.image.description="The backend engine for the Chap Modeling Platform"
LABEL org.opencontainers.image.vendor="DHIS2"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"
LABEL org.opencontainers.image.source="https://github.com/dhis2-chap/chap-core"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt upgrade -y && \
    apt install -y --no-install-recommends git curl && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
\
    useradd --create-home --shell /usr/sbin/nologin chap

WORKDIR /app

COPY --chown=root:root ./pyproject.toml ./uv.lock ./.python-version ./gunicorn.conf.py ./alembic.ini README.md ./
COPY --chown=root:root ./chap_core ./chap_core
COPY --chown=root:root ./config ./config
COPY --chown=root:root ./alembic ./alembic

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PORT=8000
ENV WORKERS=1

ENV PATH="/app/.venv/bin:$PATH"

HEALTHCHECK CMD curl --fail http://localhost:${PORT}/health || exit 1

USER chap

CMD [ \
    "gunicorn", "-c", "gunicorn.conf.py", "-k", "uvicorn.workers.UvicornWorker", "chap_core.rest_api.v1.rest_api:app", \
      "--bind 0.0.0.0:${PORT}", \
      "--workers ${WORKERS}", \
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

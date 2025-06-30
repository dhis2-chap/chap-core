# THIS DOCKERFILE RUNS THE WEB API

# Use a slim base image with Python and uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

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
    apt-get install -y --no-install-recommends git && \
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
RUN chown -R chap:chap /app
USER chap

# Ensure virtual environment is first in PATH
ENV PATH="/app/.venv/bin:$PATH"

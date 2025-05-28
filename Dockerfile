# Use slim uv-based Python image with Python 3.12
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Environment optimizations for uv and path
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Install system dependencies in one clean layer
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy pyproject + lock file first for layer caching
COPY pyproject.toml uv.lock ./

# Install only dependencies (for optimal caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy project files
COPY chap_core/ chap_core/
COPY config/ config/
COPY scripts/seed.py scripts/seed.py
COPY README.md .

# Install project itself (dev=False, project=True)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

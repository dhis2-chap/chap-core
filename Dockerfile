#THIS DOCKERFILE RUNS THE WEB API

# Use the official Python base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Set the working directory in the container
WORKDIR /app

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Install the Python dependencies
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y git

COPY ./chap_core ./chap_core
COPY ./scripts/seed.py ./scripts/seed.py
COPY ./pyproject.toml .
COPY ./uv.lock .
COPY ./README.md .

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

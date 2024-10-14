#THIS DOCKERFILE RUNS THE WEB API

# Use the official Python base image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Install the Python dependencies
RUN apt-get update
#RUN apt-get upgrade

COPY ./chap_core ./chap_core
COPY ./pyproject.toml .
COPY ./README.md .

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN pip install --upgrade pip
RUN uv pip install --system -e .

# Start the FastAPI application
#CMD chap serve & rq worker
#CMD ["chap", "serve", "&", "rq", "worker"]

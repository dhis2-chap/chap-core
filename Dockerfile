#THIS DOCKERFILE RUNS THE WEB API

# Use the official Python base image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Install the Python dependencies
RUN apt-get update

COPY ./climate_health ./climate_health
COPY ./README.md ./README.md
COPY ./setup.py ./setup.py
COPY ./HISTORY.rst ./HISTORY.rst
COPY ./README.md ./README.md
COPY ./external_models ./external_models

RUN pip install -e .

# Start the FastAPI application
#CMD chap serve & rq worker
CMD ["chap", "serve", "&", "rq", "worker"]

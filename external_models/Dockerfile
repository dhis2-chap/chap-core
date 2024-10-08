# installs chap with support for some external models
FROM ivargr/r_inla

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

# Set the working directory in the container
WORKDIR /app

# Install the Python dependencies
RUN apt-get update
#RUN apt-get upgrade

COPY ./chap_core ./chap_core
COPY ./README.md ./README.md
COPY ./setup.py ./setup.py
COPY ./HISTORY.rst ./HISTORY.rst
COPY ./README.md ./README.md
COPY ./external_models ./external_models

# get uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN pip install --upgrade pip
RUN pip install -e .

# Start the FastAPI application
CMD chap serve & rq worker
#CMD ["chap", "serve", "&", "rq", "worker"]

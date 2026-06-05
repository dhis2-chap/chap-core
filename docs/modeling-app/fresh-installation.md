# First-time Setup

Follow these steps if you're installing Chap Core for the first time.

!!! warning
    We highly recommend you to have read the [recommendation for server deployment](running-chap-on-server.md) before reading this guide.

## Prerequisites

- Docker and Docker Compose installed on your system
- Git for cloning the repository

## 1. Clone the Chap Core Repository

```console
git clone https://github.com/dhis2-chap/chap-core.git
cd chap-core
```

## 2. Checkout the Desired Version

Fetch the available versions and checkout the version you want to install. Check the releases on GitHub to see what the latest release is.

```console
# Fetch all tags
git fetch --tags

# List available versions
git tag -l

# Checkout a specific version
git checkout [VERSION] #Replace with your desired version, e.g. v1.0.18
```

For latest release go to: [https://github.com/dhis2-chap/chap-core/releases](https://github.com/dhis2-chap/chap-core/releases)

## 3. Configure Environment Variables

Copy the example environment file:

```console
cp .env.example .env
```

This creates a `.env` file with default database credentials used by Docker Compose.

!!! tip "Production deployments"
    For production, open `.env` and change at least `POSTGRES_PASSWORD` to a strong, unique value. You can optionally change `POSTGRES_USER` as well. These credentials are set permanently when the database volume is first created, so choose them before running `docker compose up` for the first time.

## 4. Start Chap Core

```console
docker compose -f compose.yml -f compose.chapkit.yml up -d
```

This command will:

- Pull all required Docker images
- Start the PostgreSQL database
- Start the Redis cache
- Start the Chap Core API server
- Start the Celery worker for background jobs
- **Automatically create and initialize your database**
- Start the bundled model services, which register themselves with Chap on startup and then appear in the modeling app — no extra configuration or rebuild needed

The Chap Core REST API will be available at `http://localhost:8000` once all services are running.

!!! note "About the model services"
    `compose.chapkit.yml` is an umbrella overlay that starts the bundled model services alongside Chap. Plain `docker compose up` (just `compose.yml`) also works and gives you Chap Core with its built-in models, but without those additional model services. `compose.yml` and `compose.ghcr.yml` are alternatives — do not stack them.

## 5. Verify the Installation

You can verify that Chap Core is running correctly by:

1. **Check the API documentation**: Visit `http://localhost:8000/docs` in your browser to see the interactive API documentation

2. **Check the health endpoint**:

```console
curl http://localhost:8000/health
```

3. **Check the available models**: confirm the models (including the bundled model services) are registered:

```console
curl http://localhost:8000/v1/crud/configured-models
```

4. **View service logs**:

```console
docker compose logs -f
```

---

## Common Operations

### Stopping Chap Core

To stop all services (pass the same `-f` flags used to start them):

```console
docker compose -f compose.yml -f compose.chapkit.yml down
```

This preserves your database data. To start again, run the same `docker compose -f compose.yml -f compose.chapkit.yml up -d` command from step 4.

### Viewing Logs

See [Troubleshooting: Viewing Logs](troubleshooting/logs.md) for how to inspect logs from the running services.

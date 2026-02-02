# First-time Setup

Follow these steps if you're installing Chap Core for the first time.

!!! warning
    We highly recommend you to have read the [recommendation for server deployment](running-chap-on-server.md) before reading this guide.

## Prerequisites

- Docker and Docker Compose installed on your system
- Git for cloning the repository

## 1. Clone the Chap Core Repository

```bash
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

## 3. Start Chap Core

```bash
docker compose up
```

This single command will:

- Pull all required Docker images
- Start the PostgreSQL database
- Start the Redis cache
- Start the Chap Core API server
- Start the Celery worker for background jobs
- **Automatically create and initialize your database**

The Chap Core REST API will be available at `http://localhost:8000` once all services are running.

## 4. Verify the Installation

You can verify that Chap Core is running correctly by:

1. **Check the API documentation**: Visit `http://localhost:8000/docs` in your browser to see the interactive API documentation

2. **Check the health endpoint**:

```bash
curl http://localhost:8000/health
```

3. **View service logs**:

```bash
docker compose logs -f
```

---

## Common Operations

### Stopping Chap Core

To stop all services:

```bash
docker compose down
```

This preserves your database data. To start again, simply run `docker compose up`.

### Viewing Logs

To view logs from all services:

```bash
docker compose logs -f
```

To view logs from a specific service:

```bash
docker compose logs -f chap
docker compose logs -f worker
docker compose logs -f postgres
```

# First-time Setup

Follow these steps if you're installing Chap Core for the first time.

!!! warning
    We highly recommend you to have read the [recommendation for server deployment](running-chap-on-server.md) before reading this guide.

## Prerequisites

- Git for cloning the repository
- Docker and Docker Compose installed on your system
- [uv](https://docs.astral.sh/uv/) for installing the `chap-admin` command, which adds models to Chap

uv installs Python tools together with the Python version they need, so the host does not need a recent Python of its own. On Linux, install it with:

```console
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

On macOS, use `brew install uv`. See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other systems. Check that it works with `uv --version`.

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
    For production, open `.env` and change at least `POSTGRES_PASSWORD` to a strong, unique value. The password is interpolated into a database URL, so keep it URL-safe -- no `@`, `:`, `/`, `?`, `#` or `%` -- or set `CHAP_DATABASE_URL` to a full percent-encoded URL instead. You can optionally change `POSTGRES_USER` as well. These credentials are set permanently when the database volume is first created, so choose them before running `docker compose up` for the first time.

    If Chap Core will be reachable from outside your internal network, also set `CHAP_API_TOKEN` to require an API token on every request. See [API Authentication](../webapi/api-authentication.md).

## 4. Start Chap Core

```console
docker compose up -d
```

This command will:

- Pull all required Docker images
- Start the PostgreSQL database
- Start the Redis cache
- Start the Chap Core API server
- Start the Celery worker for background jobs
- **Automatically create and initialize your database**

The Chap Core REST API will be available at `http://localhost:8000` once all services are running. At this point Chap has only its built-in models.

!!! note
    `compose.yml` and `compose.ghcr.yml` are alternatives — do not stack them.

## 5. Install the Models

Model services are installed from the [CHAP Model Marketplace](https://github.com/dhis2-chap/model-marketplace) with `chap-admin`, which comes with the `chap-core` package. Install the same `chap-core` version as the Chap you checked out in step 2, since `chap-admin` uses REST API endpoints of that version. Then install every model with a verified stable version from the repository directory:

```console
uv tool install chap-core==[VERSION] --python 3.13  # e.g. chap-core==1.0.18
chap-admin install-all
```

This is a manual step, run once after the first `docker compose up`. For each model it starts the model service, waits for it to register with Chap, and adds the model's verified configurations. The services are written into `compose.marketplace.yml` next to `compose.yml`. Add `--platform linux/amd64` on machines that are not AMD64, such as Apple Silicon, since some models only publish AMD64 images. If you set `CHAP_API_TOKEN` in `.env`, export it in your shell first, since `chap-admin` does not read `.env`.

From now on, include `compose.marketplace.yml` in every Docker Compose command for this deployment:

```console
docker compose -f compose.yml -f compose.marketplace.yml up -d
```

See [Running marketplace models](../chap-cli/chap-core-cli-setup.md#running-marketplace-models) for installing, updating and removing single models.

## 6. Verify the Installation

You can verify that Chap Core is running correctly by:

1. **Check the API documentation**: Visit `http://localhost:8000/docs` in your browser to see the interactive API documentation

2. **Check the health endpoint**:

```console
curl http://localhost:8000/health
```

3. **Check the available models**: confirm the installed marketplace models are listed:

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
docker compose -f compose.yml -f compose.marketplace.yml down
```

This preserves your database data. To start again, run `docker compose -f compose.yml -f compose.marketplace.yml up -d`.

### Viewing Logs

See [Troubleshooting: Viewing Logs](troubleshooting/logs.md) for how to inspect logs from the running services.

# Updating to a New Version

Follow these steps if you already have Chap Core installed and want to update to a newer version.

!!! warning
    We highly recommend you to have read the [recommendation for server deployment](running-chap-on-server.md) before reading this guide.

## Prerequisites

- Docker and Docker Compose installed on your system
- Git for cloning the repository

## 1. Backup Your Database (Recommended)

**Important:** Before upgrading, create a backup of your database to prevent data loss in case of issues.

```console
# Create a backup of the PostgreSQL database
docker compose exec -T postgres pg_dump -U ${POSTGRES_USER} chap_core > backup_$(date +%Y%m%d_%H%M%S).sql
```

## 2. Update the Repository

```console
# Navigate to your chap-core directory
cd chap-core

# Fetch the latest tags and updates
git fetch --tags

# List available versions
git tag -l

# Checkout the new version you want to upgrade to
git checkout [VERSION] #Replace with your desired version, e.g. v1.0.18
```

For latest release go to: [https://github.com/dhis2-chap/chap-core/releases](https://github.com/dhis2-chap/chap-core/releases)

!!! note "New in v1.1.5: Environment file required"
    Starting from version 1.1.5, a `.env` file is required. If you don't already have one, copy it from the example:

        cp .env.example .env

    For production deployments, edit `.env` to set secure values for `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`.

!!! warning "Upgrading to v1.1.5 with an existing database"
    Versions before 1.1.5 used hard-coded PostgreSQL credentials (`root` / `thisisnotgoingtobeexposed`). The new `.env.example` defaults to different values (`chap` / `chap`). If you have an existing database created with the old credentials, copying `.env.example` as-is will cause a connection failure because PostgreSQL keeps the credentials that were set when the volume was first created.

    To keep your existing database working, set the **old** credentials in your `.env` file:

        POSTGRES_USER=root
        POSTGRES_PASSWORD=thisisnotgoingtobeexposed
        POSTGRES_DB=chap_core

!!! note "New in v1.2.0: Runs volume target changed"
    In version 1.2.0, the runs volume target moved from `/app/runs` to `/data/runs`, controlled by the new `CHAP_RUNS_DIR` environment variable set in `compose.yml`. Existing runs data stored in the Docker volume will be automatically available at the new path since Docker volumes are path-independent, so no manual migration is needed.

## 3. Upgrade Chap Core

!!! warning "Pass the same `-f` flags you started with"
    Docker Compose has no memory of the overlay files you used last time. If you installed marketplace models with `chap-admin` (as [First-time Setup](fresh-installation.md) instructs) and then upgrade with a bare `docker compose up`, the model services are silently left out and their models stop working.

    The commands below assume `compose.yml` with the `compose.marketplace.yml` file that `chap-admin` writes. Adjust them to match how you started Chap — see the [overlay reference](../webapi/docker-compose-doc.md#compose-file-reference). If you use a `compose.override.yml` file, add `-f compose.override.yml` to every command below — Compose only picks that file up on its own when no `-f` flag is passed at all, so with the flags below it would otherwise be dropped and its services removed on upgrade.

!!! note "Deployments started with `compose.chapkit.yml`"
    Earlier versions started the EWARS model service with the `compose.chapkit.yml` overlay, and Chap created its configurations when the service registered. Chap no longer does that, so a service started this way cannot be run. Stop the stack with the old flags, start it without the overlay, and install the models with `chap-admin` as described in [Install the Models](fresh-installation.md#5-install-the-models):

    ```console
    docker compose -f compose.yml -f compose.chapkit.yml down
    docker compose up --build -d
    chap-admin install-all
    ```

    Install uv first if you do not have it, as described in the [prerequisites](fresh-installation.md#prerequisites). Use only `compose.yml` and `compose.marketplace.yml` from then on. Running both overlays starts two EWARS services under the same id, and Chap sends work to whichever registered last. Backtests made with the old EWARS model are kept.

```console
# Stop all containers first
docker compose -f compose.yml -f compose.marketplace.yml down

# Spin the containers up with --build to get new changes
docker compose -f compose.yml -f compose.marketplace.yml up --build -d
```

NOTE: There might be issues with cached images. If you encounter problems, try forcing a fresh pull of all images:

```console
docker compose -f compose.yml -f compose.marketplace.yml build --no-cache
docker compose -f compose.yml -f compose.marketplace.yml up -d
```

Docker compose up will:

- Pull any updated Docker images
- **Automatically migrate your database** to the new schema
- Start all services with the new version

The database migration happens automatically - you do not need to run any manual migration commands. In the compose.yml file, we pin postgres to a major version (17). Note that between upgrades, there might be minor incompatibilities, such as collation issues. Feel free to handle these by pinning the postgres version further, or handle the database separately.

If you installed models with `chap-admin`, upgrade it to the same version as Chap, so it matches the REST API it talks to:

```console
uv tool install chap-core==[VERSION] --python 3.13
```

## 4. Verify the Upgrade

Check that the upgrade was successful, by checking the health endpoint of chap locally:

```console
curl http://localhost:8000/health
```

## 5. Restore from Backup (If Needed)

If you encounter issues and need to restore from your backup:

```console
# Stop the services
docker compose -f compose.yml -f compose.marketplace.yml down

# Remove the database volume to start fresh
docker compose -f compose.yml -f compose.marketplace.yml down --volumes

# Start only the database
docker compose -f compose.yml -f compose.marketplace.yml up -d postgres

# Wait for postgres to initialize, then restore the backup

cat backup_20241023_120000.sql | docker compose exec -T postgres psql -U ${POSTGRES_USER} chap_core

# Start all services
docker compose -f compose.yml -f compose.marketplace.yml up --build
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

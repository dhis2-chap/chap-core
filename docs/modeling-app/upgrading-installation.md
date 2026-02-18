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

```console
# Stop all containers first
docker compose down

# Spin the containers up with --build to get new changes
docker compose up --build -d
```

NOTE: There might be issues with cached images. If you encounter problems, try forcing a fresh pull of all images:

```console
docker compose build --no-cache
docker compose up -d
```

Docker compose up will:

- Pull any updated Docker images
- **Automatically migrate your database** to the new schema
- Start all services with the new version

The database migration happens automatically - you do not need to run any manual migration commands. In the compose.yml file, we pin postgres to a major version (17). Note that between upgrades, there might be minor incompatibilities, such as collation issues. Feel free to handle these by pinning the postgres version further, or handle the database separately.

## 4. Verify the Upgrade

Check that the upgrade was successful, by checking the health endpoint of chap locally:

```console
curl http://localhost:8000/health
```

## 5. Restore from Backup (If Needed)

If you encounter issues and need to restore from your backup:

```console
# Stop the services
docker compose down

# Remove the database volume to start fresh
docker compose down --volumes

# Start only the database
docker compose up -d postgres

# Wait for postgres to initialize, then restore the backup

cat backup_20241023_120000.sql | docker compose exec -T postgres psql -U ${POSTGRES_USER} chap_core

# Start all services
docker compose up --build
```

---

## Common Operations

### Stopping Chap Core

To stop all services:

```console
docker compose down
```

This preserves your database data. To start again, simply run `docker compose up`.

### Viewing Logs

To view logs from all services:

```console
docker compose logs -f
```

To view logs from a specific service:

```console
docker compose logs -f chap
docker compose logs -f worker
docker compose logs -f postgres
```

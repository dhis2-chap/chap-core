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
docker compose exec -T postgres pg_dump -U root chap_core > backup_$(date +%Y%m%d_%H%M%S).sql
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

## 3. Upgrade Chap Core

```bash
# Stop all containers first
docker compose down

# Spin the containers up with --build to get new changes
docker compose up --build -d
```

NOTE: There might be issues with cached images. If you encounter problems, try forcing a fresh pull of all images:

```bash
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

```bash
curl http://localhost:8000/health
```

## 5. Restore from Backup (If Needed)

If you encounter issues and need to restore from your backup:

```bash
# Stop the services
docker compose down

# Remove the database volume to start fresh
docker compose down --volumes

# Start only the database
docker compose up -d postgres

# Wait for postgres to initialize, then restore the backup

cat backup_20241023_120000.sql | docker compose exec -T postgres psql -U root chap_core

# Start all services
docker compose up --build
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

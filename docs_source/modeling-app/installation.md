# Installing Chap Core for use with the Modeling app

This guide covers installing and running Chap Core using Docker Compose. This is necessary if you want to run the modeling app. Note: For only using chap-core command line interface to run models locally, simply installing chap with `pip install chap` is sufficient.

## Prerequisites

- Docker and Docker Compose installed on your system
- Git for cloning the repository

## Installation Steps

### 1. Clone the Chap Core Repository

```bash
git clone https://github.com/dhis2-chap/chap-core.git
cd chap-core
```

### 2. Checkout the Desired Version

Fetch the available versions and checkout the version you want to install. Check the releases on github to see what the latest release is.

```bash
# Fetch all tags
git fetch --tags

# List available versions
git tag -l

# Checkout a specific version (e.g., v1.0.17)
git checkout v1.0.17
```

To use the latest development version instead, stay on the master branch (skip the checkout step). Note that this branch is unstable and get frequent changes.

### 3. Start Chap Core

```bash
docker compose up
```

This single command will:
- Pull all required Docker images
- Start the PostgreSQL database
- Start the Redis cache
- Start the Chap Core API server
- Start the Celery worker for background jobs
- **Automatically upgrade and migrate your database** to the latest schema

The Chap Core REST API will be available at `http://localhost:8000` once all services are running.

## Database Migrations

**Important:** When you upgrade to a newer version of Chap Core, the database schema is **automatically migrated** when you run `docker compose up`. You do not need to manually manage database migrations or use `docker compose down --volumes`.

### Optional: Backup Your Database Before Upgrading

If you want to be extra cautious before upgrading to a new version, you can create a database backup:

```bash
# Create a backup of the PostgreSQL database
docker compose exec -T postgres pg_dump -U root chap_core > backup_$(date +%Y%m%d_%H%M%S).sql
```

To restore from a backup if needed:

```bash
# Stop the services
docker compose down

# Start only the database
docker compose up -d postgres

# Restore the backup (replace the filename with your actual backup file)
cat backup_20241023_120000.sql | docker compose exec -T postgres psql -U root chap_core

# Start all services
docker compose up
```

## Verifying the Installation

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

## Stopping Chap Core

To stop all services:

```bash
docker compose down
```

This preserves your database data. To start again, simply run `docker compose up`.

## Next Steps

- Configure Chap Core to connect with DHIS2 (see [Configure DHIS2 Modeling App](running-chap-on-server.md))
- Check the [Chap Core wiki](https://github.com/dhis2-chap/chap-core/wiki) for more information


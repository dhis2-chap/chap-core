# Installing Chap Core for use with the Modeling app

This guide covers installing and running Chap Core using Docker Compose. This is necessary if you want to run the modeling app. Note: For only using chap-core command line interface to run models locally, simply installing chap with `pip install chap` is sufficient.

## Prerequisites

- Docker and Docker Compose installed on your system
- Git for cloning the repository

---

## Fresh Installation (New Users)

Follow these steps if you're installing Chap Core for the first time.

### 1. Clone the Chap Core Repository

```bash
git clone https://github.com/dhis2-chap/chap-core.git
cd chap-core
```

### 2. Checkout the Desired Version

Fetch the available versions and checkout the version you want to install. Check the releases on GitHub to see what the latest release is.

```bash
# Fetch all tags
git fetch --tags

# List available versions
git tag -l

# Checkout a specific version (e.g., v1.0.17)
git checkout v1.0.17
```

To use the latest development version instead, stay on the master branch (skip the checkout step). Note that this branch is unstable and gets frequent changes.

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
- **Automatically create and initialize your database**

The Chap Core REST API will be available at `http://localhost:8000` once all services are running.

### 4. Verify the Installation

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

## Upgrading Existing Installation

Follow these steps if you already have Chap Core installed and want to upgrade to a newer version.

### 1. Backup Your Database (Recommended)

**Important:** Before upgrading, create a backup of your database to prevent data loss in case of issues.

```bash
# Create a backup of the PostgreSQL database
docker compose exec -T postgres pg_dump -U root chap_core > backup_$(date +%Y%m%d_%H%M%S).sql
```

### 2. Update the Repository

```bash
# Navigate to your chap-core directory
cd chap-core

# Fetch the latest tags and updates
git fetch --tags

# List available versions
git tag -l

# Checkout the new version you want to upgrade to
git checkout v1.0.18  # Replace with your desired version
```

### 3. Upgrade Chap Core

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

Docke compose up will:
- Pull any updated Docker images
- **Automatically migrate your database** to the new schema
- Start all services with the new version

The database migration happens automatically - you do not need to run any manual migration commands. In the compose.yml file, we pin postgres to a major version (17). Note that between upgrades, there might be minor incompatibilities, such as collation issues. Feel free to handle these by pinning the postgres version further, or handle the database separately.

### 4. Verify the Upgrade

Check that the upgrade was successful, by checking the health endopint of chap locally:

   ```bash
   curl http://localhost:8000/health
   ```

### 5. Restore from Backup (If Needed)

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

---

## Next Steps

- Configure Chap Core to connect with DHIS2 (see [Configure DHIS2 Modeling App](running-chap-on-server.md))
- Check the [Chap Core wiki](https://github.com/dhis2-chap/chap-core/wiki) for more information

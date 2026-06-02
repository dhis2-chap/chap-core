# Accessing the Database

Chap Core stores its data in a PostgreSQL database that runs as the `postgres` service in Docker Compose. The database port is not published to the host, so you connect to it through the running `postgres` container rather than from your host machine.

The database credentials come from your `.env` file: `POSTGRES_USER` (default `chap`), `POSTGRES_PASSWORD` (default `chap`), and `POSTGRES_DB` (default `chap_core`).

## Open a psql shell

To open an interactive `psql` session:

```console
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

Running it as `sh -c '...'` lets `psql` read the credentials from the container's own environment, so it works regardless of what you set in `.env`. From the prompt you can list the tables with `\dt`, run SQL queries, and type `\q` to exit.

## Run a one-off command

To run a single command without opening an interactive session, pass it with `-c`. For example, to list the tables:

```console
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\dt"'
```

The same works for SQL queries, for example:

```console
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT COUNT(*) FROM dataset;"'
```

## Backing up and restoring

For creating and restoring database backups with `pg_dump`, see [Backup Your Database](../upgrading-installation.md#1-backup-your-database-recommended) in the upgrade guide.

# Viewing Logs

When something goes wrong with a Docker Compose deployment of Chap Core, the logs are usually the first place to look. There are two complementary places to check.

## Service logs (`docker compose logs`)

The REST API and worker log to their container output (stdout/stderr), so use `docker compose logs` to read them.

To stream logs from all services:

```console
docker compose logs -f
```

To view logs from a specific service:

```console
docker compose logs -f chap
docker compose logs -f worker
docker compose logs -f postgres
```

The most relevant services are:

- `chap`: the Chap Core REST API
- `worker`: the Celery worker that runs the models. Check this if a model run fails
- `postgres`: the database

## Per-task log files

In addition to the service output, the worker writes a pair of log files for each task (typically a model run) into the directory given by the `CHAP_LOGS_DIR` environment variable. In the default Docker Compose setup this is `/data/logs`, backed by a named `logs` volume shared between the `chap` and `worker` services:

- `task_{task_id}.debug.txt`: full debug logs for the task (server-side only). Check this when a model run fails
- `task_{task_id}.status.txt`: user-facing progress messages for the task (also exposed via the API)

`{task_id}` is the internal Celery task id. Because these files live inside the container's volume rather than in the repository on the host, read them through the running container, for example:

```console
# List the task log files
docker compose exec worker ls /data/logs

# View the debug log for a specific task
docker compose exec worker cat /data/logs/task_<task_id>.debug.txt
```

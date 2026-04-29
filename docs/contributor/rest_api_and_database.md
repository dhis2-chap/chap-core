# REST API and database architecture

This guide explains how the CHAP Core REST API and database layers are structured,
how they connect, and how to work with them. It is aimed at developers who are new
to the codebase.

## High-level overview

```
                     +-----------+
                     |  FastAPI   |
                     |   app.py   |
                     +-----+-----+
                           |
          +----------------+----------------+
          |                |                |
    common_routes     /v1 router       /v2 router
    (health, info)         |                |
                     +-----+-----+     services.py
                     |           |     (chapkit registry)
                  crud.py   analytics.py
                  jobs.py   visualization.py
```

The application is a **FastAPI** server defined in `rest_api/app.py`.
It mounts three groups of routes:

- **Common routes** (`common_routes.py`) -- health check and system info at the root level.
- **v1** (`rest_api/v1/`) -- the main API used by the Modeling App frontend.
- **v2** (`rest_api/v2/`) -- the service registry for chapkit model services.

Long-running operations (backtests, predictions, dataset imports) are executed
asynchronously via **Celery** with a **Redis** broker. The REST endpoints return a
job ID that clients poll via the `/v1/jobs` endpoints.

Persistent data is stored in **PostgreSQL** using **SQLModel** (which builds on
SQLAlchemy and Pydantic).


## Infrastructure components

| Component   | Role                                       | Config env var        |
|-------------|--------------------------------------------|-----------------------|
| PostgreSQL  | Persistent storage for all domain data     | `CHAP_DATABASE_URL`   |
| Redis       | Celery broker/backend + v2 service registry| `CELERY_BROKER`       |
| Celery      | Async task execution                       | (uses Redis URL)      |


## The v1 API

The v1 router (`rest_api/v1/rest_api.py`) includes four sub-routers:

### crud.py (`/v1/crud/...`)

Standard CRUD endpoints for the core domain objects:

- **Backtests** -- list, get, create, update, delete evaluations.
- **Predictions** -- list, get, delete predictions.
- **Datasets** -- list, get, create (JSON or CSV), export as CSV/DataFrame, delete.
- **Model templates** -- list all templates (also triggers chapkit sync).
- **Configured models** -- list, create, soft-delete configured models.
- **Debug** -- create and get debug entries.

Creating a backtest (`POST /v1/crud/backtests`) queues a Celery job and returns a
`JobResponse` with the task ID. The actual work happens in `db_worker_functions.py`,
executed by the Celery worker.

### analytics.py (`/v1/analytics/...`)

Higher-level endpoints used by the Modeling App:

- `POST /make-dataset` -- validate, harmonize and import a dataset.
- `POST /create-backtest` -- create a backtest from an existing dataset.
- `POST /create-backtest-with-data/` -- validate data, create dataset, then run backtest. This is the main endpoint used by the Modeling App. Supports a `dryRun` query param for validation only.
- `POST /make-prediction` -- validate data and run a prediction.
- `GET /evaluation-entry` -- return quantile-based forecast data for a backtest.
- `GET /prediction-entry/{id}` -- return quantile-based forecast data for a prediction.
- `GET /actualCases/{id}` -- return observed disease cases for a backtest's dataset.
- `GET /compatible-backtests/{id}` -- find backtests compatible for comparison.
- `GET /backtest-overlap/{id1}/{id2}` -- find overlapping org units and periods.
- `GET /data-sources` -- return the list of available climate data sources.

### jobs.py (`/v1/jobs/...`)

Monitor and manage async Celery jobs:

- `GET /v1/jobs` -- list all jobs, with optional filters for IDs, status, and type.
- `GET /v1/jobs/{id}` -- get job status (PENDING, STARTED, SUCCESS, FAILURE, REVOKED).
- `DELETE /v1/jobs/{id}` -- delete a completed job's metadata.
- `POST /v1/jobs/{id}/cancel` -- cancel a running job.
- `GET /v1/jobs/{id}/logs` -- get user-facing status logs for a job.
- `GET /v1/jobs/{id}/database_result` -- get the database ID produced by a completed job.
- `GET /v1/jobs/{id}/prediction_result` -- get prediction results.
- `GET /v1/jobs/{id}/evaluation_result` -- get evaluation results.

### visualization.py (`/v1/visualization/...`)

Generates plots and metrics charts from backtest results.


## The v2 API

The v2 API currently only contains the **service registry** for chapkit model services.

### Service lifecycle

1. A chapkit container starts and calls `POST /v2/services/$register` with its URL and model info.
2. The orchestrator stores the registration in Redis with a TTL (default 30 seconds).
3. The chapkit container sends periodic `PUT /v2/services/{id}/$ping` requests to stay registered.
4. If pings stop, Redis automatically expires the registration.

The registration endpoint also eagerly syncs the chapkit service into the PostgreSQL
database (model templates and configured models) so the v1 CRUD endpoints can serve
them immediately.

### Authentication

Registration and ping endpoints require a service key via the `verify_service_key`
dependency.


## Database layer

### Engine and session setup

The database engine is created at module import time in `database/database.py`.
It reads `CHAP_DATABASE_URL` from the environment and retries connections up to 30
times (to handle container startup ordering in Docker Compose).

If the environment variable is not set, the engine is `None` and database operations
will not work. This is intentional -- CLI commands that don't need the database can
still import the module.

### Session management

There are two session patterns used in the codebase:

1. **FastAPI dependency injection** -- `get_session()` in `dependencies.py` yields a
   plain `sqlmodel.Session`. Used by most REST endpoints.

2. **SessionWrapper** -- a context manager that wraps a `Session` and adds higher-level
   data access methods (adding datasets, model templates, configured models, etc.).
   Used by the Celery worker (`celery_run_with_session`) and by some REST endpoints that
   need complex operations.

### Database tables

All table models inherit from `DBModel` (defined in `base_tables.py`), which extends
`SQLModel` with automatic **camelCase aliasing** via Pydantic's `alias_generator`.
This means snake_case field names in Python are automatically converted to camelCase
in JSON API responses.

A class becomes a database table when it has `table=True` in its class definition.

The tables are spread across several files:

| File | Tables | Purpose |
|------|--------|---------|
| `base_tables.py` | `DBModel` (base) | Base class with camelCase config |
| `dataset_tables.py` | `DataSet`, `Observation` | Imported health/climate datasets |
| `tables.py` | `BackTest`, `Prediction`, `BackTestForecast`, `PredictionSamplesEntry`, `BackTestMetric` | Evaluation and prediction results |
| `model_templates_and_config_tables.py` | `ModelTemplateDB`, `ConfiguredModelDB` | Model definitions and configurations |
| `model_spec_tables.py` | `ModelSpecRead` | Read model for backwards-compatible API responses |
| `debug.py` | `DebugEntry` | Debug/diagnostic entries |

### Key relationships

```
ModelTemplateDB  1--*  ConfiguredModelDB
DataSet          1--*  Observation
DataSet          1--*  BackTest
DataSet          1--*  Prediction
BackTest         1--*  BackTestForecast
BackTest         1--*  BackTestMetric
Prediction       1--*  PredictionSamplesEntry
ConfiguredModelDB 1--* BackTest
ConfiguredModelDB 1--* Prediction
```

### Read models and response types

The codebase uses a pattern where table models have companion "read" classes for API
responses. For example:

- `BackTest` (table) -> `BackTestRead` (API response, includes nested dataset/model info)
- `Prediction` (table) -> `PredictionInfo` (API response)
- `DataSet` (table) -> `DataSetInfo` (list response), `DataSetWithObservations` (detail response)

These read models are defined near their table models and often use inheritance.
The `DBModel.get_read_class()` and `get_create_class()` methods can auto-generate
simple read/create variants, but most models define their read classes explicitly.

### Database migrations

The system uses a hybrid migration approach:

1. **Generic migration** (`_run_generic_migration`) -- scans SQLModel metadata for
   missing columns and adds them with appropriate defaults. Handles simple schema
   evolution automatically.
2. **Alembic** (`_run_alembic_migrations`) -- runs standard Alembic migrations for
   more complex schema changes. Config is in `alembic.ini` at the project root.

Both run during `create_db_and_tables()`, which is called at application startup.

### Model seeding

After migrations, `seed_configured_models_from_config_dir()` seeds the database with
model templates and configured models from YAML config files.


## Async job processing (Celery)

### How it works

1. A REST endpoint calls `worker.queue_db(func, *args, **kwargs)` on a `CeleryPool`.
2. This serializes the function and arguments (using pickle) and sends them to the
   Redis broker via `celery_run_with_session.delay()`.
3. The Celery worker picks up the task, creates a `SessionWrapper` with a fresh DB
   engine, and calls the function with the session injected.
4. Job metadata (status, timestamps, results) is stored in Redis hashes (`job_meta:{task_id}`).

### TrackedTask

The `TrackedTask` base class (`celery_tasks.py`) adds:

- Per-task **log files** (both debug and user-facing status logs).
- **Redis metadata** updates on task start, success, and failure.
- Traceback capture on failure.

### Job types

The `JobType` enum defines the canonical job types:

- `EVALUATION_LEGACY` ("create_backtest") -- backtest from existing dataset
- `EVALUATION` ("create_backtest_from_data") -- backtest with inline data
- `PREDICTION` ("create_prediction") -- prediction
- `DATASET` ("create_dataset") -- dataset import

These string values are the contract with the Modeling App frontend.

### Worker functions

The actual business logic for async jobs lives in `db_worker_functions.py`. These
functions receive a `SessionWrapper` (injected by `celery_run_with_session`) and
perform the database operations.


## Chapkit service integration

Chapkit is an external model service framework. The integration works as follows:

1. Chapkit containers register via `POST /v2/services/$register`.
2. The `Orchestrator` stores registrations in Redis with TTL-based expiration.
3. When `GET /v1/crud/model-templates` is called, `_sync_live_chapkit_services()`
   queries the orchestrator and upserts model templates/configured models into
   PostgreSQL.
4. Stale chapkit templates (whose services are no longer live) are archived.
5. When running a backtest or prediction with a chapkit model, `SessionWrapper.get_configured_model_with_code()` resolves the live service URL from the
   orchestrator (Redis) and falls back to the stored URL if unavailable.


## camelCase conversion

All `DBModel` subclasses use `alias_generator=to_camel` via Pydantic's `ConfigDict`.
This means:

- Python code uses `snake_case` field names.
- JSON serialization uses `camelCase` (because `response_model_by_alias=True` on endpoints).
- API consumers receive camelCase. Path/query parameters that correspond to camelCase
  fields use explicit `alias="camelCase"` annotations in endpoint signatures.

The crud router uses a `router_get = partial(router.get, response_model_by_alias=True)`
shortcut to apply this to all GET endpoints.


## How to add a new endpoint

1. Decide which router it belongs to (crud, analytics, visualization, jobs).
2. Define Pydantic request/response models. Extend `DBModel` if you want camelCase
   aliases.
3. Add the endpoint function to the appropriate router file.
4. If the operation is long-running, queue it as a Celery task via `worker.queue_db()`
   and return a `JobResponse`.
5. For new database operations, add methods to `SessionWrapper` or work with
   the `Session` directly in the endpoint.


## How to add a new database table

1. Define the table model in the appropriate file under `database/`, inheriting from
   `DBModel` with `table=True`.
2. Define relationships using SQLModel's `Relationship()`.
3. Optionally define companion read/create models for the API.
4. The generic migration system will automatically add the new table on startup.
   For column additions to existing tables, it also handles those. For more complex
   changes (renaming, type changes), create an Alembic migration.


## File reference

| File | Description |
|------|-------------|
| `rest_api/app.py` | FastAPI app, CORS, global exception handler, router mounting |
| `rest_api/common_routes.py` | `/health`, `/system/info` endpoints |
| `rest_api/v1/rest_api.py` | v1 router aggregation |
| `rest_api/v1/routers/crud.py` | CRUD endpoints + chapkit sync logic |
| `rest_api/v1/routers/analytics.py` | Dataset/backtest/prediction creation endpoints |
| `rest_api/v1/routers/visualization.py` | Plot and chart generation |
| `rest_api/v1/routers/dependencies.py` | FastAPI dependency injection (session, settings) |
| `rest_api/v1/jobs.py` | Job monitoring and management endpoints |
| `rest_api/v2/routers/services.py` | Chapkit service registration/discovery |
| `rest_api/v2/dependencies.py` | v2 dependency injection (orchestrator, Redis) |
| `rest_api/services/orchestrator.py` | Redis-backed service registry |
| `rest_api/services/schemas.py` | Pydantic schemas for service registration |
| `rest_api/celery_tasks.py` | Celery app, TrackedTask, CeleryPool, job metadata |
| `rest_api/data_models.py` | Shared Pydantic request/response models |
| `rest_api/db_worker_functions.py` | Business logic for async Celery jobs |
| `rest_api/worker_functions.py` | WorkerConfig and related utilities |
| `database/database.py` | Engine creation, SessionWrapper, migrations |
| `database/base_tables.py` | DBModel base class with camelCase config |
| `database/tables.py` | BackTest, Prediction, forecast tables |
| `database/dataset_tables.py` | DataSet, Observation tables |
| `database/model_templates_and_config_tables.py` | ModelTemplateDB, ConfiguredModelDB |
| `database/model_spec_tables.py` | ModelSpecRead (backwards-compatible read model) |
| `database/debug.py` | DebugEntry table |

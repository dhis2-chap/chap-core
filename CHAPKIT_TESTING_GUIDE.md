# Chapkit Integration Testing Guide

Step-by-step instructions for manually testing chapkit integration with chap-core.

## Prerequisites

- chap-core running locally (API + worker + Redis + PostgreSQL)
- A chapkit model service (either local or remote)

## 1. Start chap-core stack

```bash
docker compose up -d
```

Verify it's running:

```bash
curl http://localhost:8000/v1/crud/model-templates
```

## 2. Start a chapkit model

### Option A: Use the test fixture model

```bash
cd tests/fixtures/chapkit_test_model
uv sync
SERVICEKIT_ORCHESTRATOR_URL=http://localhost:8000/v2/services/\$register \
    uv run uvicorn main:app --port 8080
```

### Option B: Use a real chapkit model

```bash
chapkit init my-model
cd my-model
uv sync
SERVICEKIT_ORCHESTRATOR_URL=http://localhost:8000/v2/services/\$register \
    uv run fastapi dev
```

### Option C: Use an already-deployed model

If a chapkit model is already running (e.g. at `http://192.168.1.100:8000`), register it manually:

```bash
# Get the service info
curl http://192.168.1.100:8000/api/v1/info

# Register with chap-core
curl -X POST http://localhost:8000/v2/services/\$register \
  -H "Content-Type: application/json" \
  -d '{"url": "http://192.168.1.100:8000", "info": <paste info JSON here>}'
```

## 3. Verify self-registration

Once the chapkit service is running with `SERVICEKIT_ORCHESTRATOR_URL` set, it auto-registers. Verify:

```bash
# Check v2 service registry
curl http://localhost:8000/v2/services | jq

# Check model templates (triggers sync from registry to DB)
curl http://localhost:8000/v1/crud/model-templates | jq '.[].name'

# Verify health status
curl http://localhost:8000/v1/crud/model-templates | jq '.[] | {name, healthStatus}'

# Check configured models were created
curl http://localhost:8000/v1/crud/configured-models | jq '.[].name'
```

## 4. Test via CLI (quickest end-to-end test)

Run an evaluation directly against the chapkit service URL:

```bash
chap eval \
    --model-name http://localhost:8080 \
    --dataset-csv example_data/vietnam_monthly.csv \
    --output-file /tmp/chapkit_eval_test.nc \
    --backtest-params.n-splits 2
```

The `--model-name` URL is auto-detected as a chapkit service (probes `/api/v1/info`).

## 5. Test via REST API (full stack)

### Create a dataset

```bash
# Upload dataset (uses example data)
curl -X POST http://localhost:8000/v1/analytics/make-dataset \
  -H "Content-Type: application/json" \
  -d @example_data/create-dataset-request.json

# Note the job ID from the response, then poll:
curl http://localhost:8000/v1/jobs/<job-id>

# Once status is "success", get the dataset ID:
curl http://localhost:8000/v1/jobs/<job-id>/database_result/
```

### Create a backtest

```bash
curl -X POST http://localhost:8000/v1/crud/backtests/ \
  -H "Content-Type: application/json" \
  -d '{
    "modelId": "chapkit-test-model",
    "datasetId": <dataset-id>,
    "name": "chapkit integration test"
  }'

# Poll job until complete:
curl http://localhost:8000/v1/jobs/<job-id>

# Fetch results:
curl http://localhost:8000/v1/crud/backtests/<backtest-id>/full | jq
```

### Or use the all-in-one endpoint

```bash
curl -X POST http://localhost:8000/v1/analytics/create-backtest-with-data/ \
  -H "Content-Type: application/json" \
  -d @example_data/create-backtest-with-data.json
```

(Edit the JSON to set `"modelId": "chapkit-test-model"` first.)

## 6. Test via modeling app (full UI flow)

1. Open the DHIS2 modeling app
2. Navigate to model selection
3. The chapkit model should appear with "live" status
4. Select it and configure a backtest
5. Run and verify results

## 7. Test deregistration

Stop the chapkit service (Ctrl+C). After the Redis TTL expires (default 30s):

```bash
# Template should become archived
curl http://localhost:8000/v1/crud/model-templates | jq '.[] | select(.name == "chapkit-test-model") | {archived, healthStatus}'
```

Restart the service to verify it un-archives automatically.

## Automated Tests

### Unit tests (mocked, fast -- included in `make test`)

```bash
pytest tests/external/test_chapkit_integration.py -v
```

Covers: REST API wrapper serialization, config conversion (`MLServiceInfo` to `ModelTemplateConfig`), geo serialization, typed responses. All HTTP calls are mocked.

### Self-registration tests (fakeredis, fast -- included in `make test`)

```bash
pytest tests/integration/rest_api/test_chapkit_self_registration.py -v
```

Covers: service registration via v2 Orchestrator, health status tracking, default and multi-config model creation, config sync, archival on deregistration (including when config sync failed), re-registration with metadata updates, graceful Redis failure. Uses fakeredis and in-memory SQLite (no real services).

### End-to-end tests (real chapkit subprocess, slow -- requires `--run-slow`)

```bash
pytest tests/integration/test_chapkit_e2e.py -v --run-slow
```

Starts a real chapkit service from `tests/fixtures/chapkit_test_model/` as a subprocess.

| Test | What it verifies |
|------|-----------------|
| `test_chapkit_service_is_healthy` | Service starts and responds to `/health` |
| `test_chapkit_service_info` | Service metadata matches (`id`, `period_type`) |
| `test_chapkit_eval_cli` | Full `chap eval` CLI against live service with example data |
| `test_chapkit_backtest_via_worker_function` | DB backtest flow: seeds model template + configured model from live service info, loads dataset from CSV, runs `run_backtest()` directly (same path as POST /v1/crud/backtests/ minus Celery) |

### Test coverage summary

| Flow | Automated? | Test location |
|------|-----------|---------------|
| REST API wrapper (train/predict HTTP calls) | Yes (mocked) | `test_chapkit_integration.py` |
| Self-registration + health status + archival | Yes (fakeredis) | `test_chapkit_self_registration.py` |
| CLI eval against live service | Yes (subprocess) | `test_chapkit_e2e.py` |
| DB backtest against live service | Yes (direct call) | `test_chapkit_e2e.py` |
| Multi-config sync (2+ configs per service) | Yes (fakeredis) | `test_chapkit_self_registration.py` |
| Re-registration with metadata changes | Yes (fakeredis) | `test_chapkit_self_registration.py` |
| REST API backtest via Celery | No (needs Redis + Celery worker) | Manual or Docker compose |
| Modeling app UI flow | No | Manual only |

## How Multiple Configs Work

A single chapkit service can expose multiple named configurations via `/api/v1/configs`. When chap-core discovers a service for the first time:

1. It creates one `ModelTemplateDB` from the service metadata (name = service id)
2. It calls `list_configs()` on the service
3. For each config returned, it creates a `ConfiguredModelDB`:
   - Default config (or empty list): name = `{template_name}` (e.g. `my-model`)
   - Named configs: name = `{template_name}:{config_name}` (e.g. `my-model:config-a`)
4. Config sync only happens on first discovery. Subsequent syncs skip if configured models already exist.

The configured model in chap-core stores an **empty reference** -- the actual config values (hyperparameters, etc.) live in the chapkit service. When a backtest runs, chap-core passes the config name to chapkit, which applies its own stored values.

## V1 API Limitations

The chapkit integration is built on top of the existing v1 REST API. This works but has some inherent limitations:

| Limitation | Details |
|-----------|---------|
| **Sync on every GET** | `GET /v1/crud/model-templates` queries Redis and syncs templates on every request, adding latency when Redis or services are slow |
| **Config values are opaque** | Configured model configs are stored as empty references in chap-core; actual values live in the chapkit service |
| **User options not synced** | The chapkit config schema (`user_options`) is not fetched during template sync; it's resolved lazily when needed |
| **No artifact references** | Backtest results don't store chapkit `artifact_id` -- these are transient during train/predict and discarded |
| **Celery required for backtests** | REST API backtests go through Celery, so both a worker and Redis must be running |
| **No migration for new fields** | `requires_geo` and `documentation_url` exist on the Python model but not in the DB schema; they are re-synced from the live service on every request and will be missing on old databases |

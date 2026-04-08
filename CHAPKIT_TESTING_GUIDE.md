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

```bash
# Unit tests (mocked, fast)
pytest tests/external/test_chapkit_integration.py -v

# Self-registration tests (fakeredis, fast)
pytest tests/integration/rest_api/test_chapkit_self_registration.py -v

# End-to-end tests (real chapkit subprocess, slow)
pytest tests/integration/test_chapkit_e2e.py -v --run-slow
```

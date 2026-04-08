# Chapkit Manual Testing Session

## Environment
- DHIS2 at http://localhost:8080
- Modeling app at http://localhost:8080/apps/dhis2-chapmodeling-app
- chap-core at http://localhost:8000 (Docker: chap, worker, postgres, redis)

## Steps to Reproduce

### 1. Verify chap-core stack is running

```bash
docker ps --format '{{.Names}} {{.Ports}}'
# Expected: chap, worker, postgres, redis containers running
```

### 2. Check initial state - no chapkit services registered

```bash
curl -s http://localhost:8000/v2/services | jq
# Returns: {"count":0,"services":[]}

curl -s http://localhost:8000/v1/crud/model-templates | jq '.[].name'
# Returns: 10 pre-seeded models, all with healthStatus: null
```

### 3. Start chapkit test model with registration

```bash
cd tests/fixtures/chapkit_test_model
uv sync
SERVICEKIT_ORCHESTRATOR_URL=http://localhost:8000/v2/services/\$register \
    uv run uvicorn main:app --host 127.0.0.1 --port 8090
```

### 4. Verify registration

```bash
curl -s http://localhost:8000/v2/services | jq
# Should show chapkit-test-model registered

curl -s http://localhost:8000/v1/crud/model-templates | jq '.[] | select(.name == "chapkit-test-model")'
# Should show template with healthStatus: "live"
```

### 5. Check in modeling app

Navigate to Models page: http://localhost:8080/apps/dhis2-chapmodeling-app#/models
The chapkit-test-model should appear in the list.

## Findings

### Finding 1: No chapkit model visible in modeling app (initial)
- **Status**: Expected - no chapkit service was running
- **Cause**: The v2 service registry was empty (`count: 0`)
- **Fix**: Start the test fixture model with `SERVICEKIT_ORCHESTRATOR_URL` pointing to chap-core

### Finding 2: Chapkit template registered but no configured model created
After starting the test fixture model with registration:

```bash
SERVICEKIT_ORCHESTRATOR_URL=http://localhost:8000/v2/services/\$register \
    uv run uvicorn main:app --host 127.0.0.1 --port 8090
```

- Template appears in `GET /model-templates` with `healthStatus: "live"`
- But no configured model was created
- The model does NOT appear in the modeling app's Models page (which shows configured models)

**Root cause 1: Wrong service URL registered**
- Service auto-detected hostname as `mlaptop.local` and defaulted port to 8000
- Actual service is at `127.0.0.1:8090`
- Docker container tried `connect_tcp host='mlaptop.local' port=8000` and failed
- Fix: Set `SERVICEKIT_HOST` and `SERVICEKIT_PORT` explicitly

**Root cause 2: `/api/v1/configs` endpoint returns 404**
- `CHAPKitRestAPIWrapper.list_configs()` calls `/api/v1/configs` on the chapkit service
- The chapkit test model doesn't have this endpoint (it's at a different path)
- Docker logs: `GET /api/v1/configs HTTP/1.1 404`
- This causes `_sync_chapkit_configured_models` to return early without creating any configured model

### Finding 3: Service URL needs explicit host/port for Docker networking
When chap-core runs in Docker and the chapkit service runs on the host:
- The auto-detected hostname (e.g. `mlaptop.local`) may not be resolvable from inside Docker
- `SERVICEKIT_HOST` env var was NOT picked up by servicekit (possible bug - `host_source=auto-detected` in logs despite env being set)
- `SERVICEKIT_PORT` env var WAS picked up correctly
- **Workaround**: Register manually with correct URL:
  ```bash
  curl -X POST http://localhost:8000/v2/services/\$register \
    -H "Content-Type: application/json" \
    -d '{"url": "http://host.docker.internal:8090", "info": <service info>}'
  ```

### Finding 4: Configured model created after manual registration with correct URL
- After manual registration with `http://host.docker.internal:8090`, chap-core's Docker container could reach the service
- `list_configs()` returned `[]` (empty), so a default configured model was created
- `curl http://localhost:8000/v1/crud/configured-models` shows `chapkit-test-model` (12 total)
- BUT the modeling app at `#/models` still shows only 11 items
- **Cause: browser/proxy caching** - hard refresh resolved it, model appears correctly

### Finding 5: SERVICEKIT_HOST env var not respected by servicekit
- Logs show `host_source=auto-detected` even when `SERVICEKIT_HOST=host.docker.internal` is exported
- This is a **servicekit bug** - the env var should override auto-detection
- Port env var (`SERVICEKIT_PORT`) works correctly (`port_source=env:SERVICEKIT_PORT`)

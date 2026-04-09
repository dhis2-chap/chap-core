# Chapkit Manual Testing via Modeling App

Step-by-step guide for testing chapkit integration through the DHIS2 modeling app.

## Prerequisites

- Docker running with DHIS2 (port 8080) and chap-core stack (chap, worker, postgres, redis on port 8000)
- chap-core branch with chapkit self-registration feature

## Step 1: Verify stack is running

```bash
docker ps --format '{{.Names}} {{.Ports}}'
# Expected: chap (8000), worker, postgres, redis, dhis2 (8080)
```

## Step 2: Configure DHIS2 route (first time / after Docker restart)

The modeling app needs a route to reach chap-core. If you see "Could not connect to CHAP":

1. Open http://localhost:8080/apps/dhis2-chapmodeling-app#/settings
2. Under "Route configuration", click the edit button (pencil icon)
3. Set URL to `http://host.docker.internal:8000/**`
4. Click Save
5. Also click "Increase to 30 seconds" for the response timeout
6. Verify "Status: Connected" and CHAP version appears

## Step 3: Start the chapkit test model

```bash
cd tests/fixtures/chapkit_test_model
uv sync
SERVICEKIT_ORCHESTRATOR_URL=http://localhost:8000/v2/services/\$register \
SERVICEKIT_HOST=host.docker.internal \
SERVICEKIT_PORT=8090 \
    uv run uvicorn main:app --host 0.0.0.0 --port 8090
```

`SERVICEKIT_HOST=host.docker.internal` ensures the service registers with a URL that chap-core's Docker container can resolve (requires chapkit >= 0.16.7).

## Step 4: Verify registration

```bash
# Check v2 registry
curl -s http://localhost:8000/v2/services | jq '.services[].id'
# Expected: "chapkit-test-model"

# Trigger sync and check template
curl -s http://localhost:8000/v1/crud/model-templates | jq '.[] | select(.name == "chapkit-test-model") | {name, healthStatus, requiredCovariates}'
# Expected: healthStatus "live", requiredCovariates ["population", "rainfall", "mean_temperature"]

# Check configured model was created
curl -s http://localhost:8000/v1/crud/configured-models | jq '.[].name' | grep chapkit
# Expected: "chapkit-test-model"
```

## Step 5: Run an evaluation with EWARS (reference)

This shows the full flow with a known working model.

1. Navigate to http://localhost:8080/apps/dhis2-chapmodeling-app#/evaluate/new
2. Fill in:
   - **Name**: EWARS Chapkit Test
   - **Period type**: Monthly
   - **From period**: 2023-01
   - **To period**: 2024-12
   - **Organisation units**: Select level "Province" (18 provinces, Lao PDR)
   - **Model**: Click "Select model" > choose "CHAP-EWARS Model" > Confirm
3. Click "Configure sources" and map:
   - Disease cases -> "Dengue Cases (Any) - Monthly"
   - Population -> "Population by year"
   - Rainfall -> "Precipitation (CHIRPS)"
   - Mean temperature -> "Air temperature (ERA5-Land)"
4. Click Save
5. Click "Start dry run" - should say "Valid import: All 18 locations"
6. Close dry run dialog
7. Click "Start import" - redirects to Jobs page
8. Wait ~5-6 minutes for job to complete (status: Running -> Successful)

## Step 6: Run an evaluation with Chapkit Test Model

Same flow as Step 5 but select "Chapkit Test Model" instead of CHAP-EWARS:

1. Navigate to http://localhost:8080/apps/dhis2-chapmodeling-app#/evaluate/new
2. Fill in same parameters (name, periods, org units)
3. Click "Select model" > choose "Chapkit Test Model" > Confirm
4. Click "Configure sources" and map same data items:
   - Disease cases -> "Dengue Cases (Any) - Monthly"
   - Population -> "Population by year"
   - Rainfall -> "Precipitation (CHIRPS)"
   - Mean temperature -> "Air temperature (ERA5-Land)"
5. Save, dry run, start import
6. Monitor job on Jobs page

## Known Issues

### Browser caching
After registering a new chapkit service, the modeling app may not show it immediately. Hard refresh (Cmd+Shift+R) resolves this.

### DHIS2 route configuration
The route URL must use `host.docker.internal` when DHIS2 runs in Docker and chap-core is on the host or a different Docker network. The default `http://chap-core:8000/**` only works when both are on the same Docker compose network.

### SERVICEKIT_HOST env var (fixed in chapkit 0.16.7)
Previously the `SERVICEKIT_HOST` env var was ignored. Fixed in servicekit 0.8.2 / chapkit 0.16.7. Now `SERVICEKIT_HOST=host.docker.internal` works correctly.

### Data item mapping
The modeling app maps model covariates to DHIS2 data elements/indicators. The covariate names in `required_covariates` must match what the app expects:
- `disease_cases` (target) -> "Dengue Cases (Any) - Monthly"
- `population` -> "Population by year"
- `rainfall` -> "Precipitation (CHIRPS)"
- `mean_temperature` -> "Air temperature (ERA5-Land)"

### Multiple configured models per chapkit template
Creating a second configured model from a chapkit template via `POST /configured-models` works correctly:
- `uses_chapkit` is inherited from the template
- The API returns both models with correct covariates
- However, the modeling app's model selection dialog shows one card per template, not per configured model
- The variant model is visible on the Models page but not selectable in the evaluation form
- This is a frontend limitation, not a backend issue

### Verified evaluation results
Both EWARS and Chapkit Test Model evaluations completed successfully:
- EWARS: ~2 minutes (full INLA model)
- Chapkit Test Model: ~30 seconds (returns mean of disease_cases)

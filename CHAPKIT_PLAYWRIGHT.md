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
SERVICEKIT_PORT=8090 \
    uv run uvicorn main:app --host 0.0.0.0 --port 8090
```

The `host="host.docker.internal"` is hardcoded in main.py as a workaround for the servicekit SERVICEKIT_HOST env var bug (see SERVICEKIT_BUGS.md).

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

### SERVICEKIT_HOST env var ignored (servicekit bug)
The `SERVICEKIT_HOST` environment variable is not respected by servicekit - it always auto-detects the hostname. Workaround: pass `host="host.docker.internal"` explicitly in `.with_registration()`. See SERVICEKIT_BUGS.md.

### Zero covariates warning
When a model has `required_covariates: []`, the dataset configuration shows "All data items mapped" but also "Please map all model covariates to valid data items". This is a frontend display bug - the form works correctly with just `disease_cases` mapped.

### Data item mapping
The modeling app maps model covariates to DHIS2 data elements/indicators. The covariate names in `required_covariates` must match what the app expects:
- `population` -> "Population by year"
- `rainfall` -> "Precipitation (CHIRPS)"
- `mean_temperature` -> "Air temperature (ERA5-Land)"
- `disease_cases` (target) -> "Dengue Cases (Any) - Monthly"

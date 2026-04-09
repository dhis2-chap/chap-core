# Chapkit Manual Testing via Modeling App

Step-by-step guide for testing chapkit integration through the DHIS2 modeling app.

## Prerequisites

- Docker running with DHIS2 (port 8080) and chap-core stack (chap, worker, postgres, redis on port 8000)
- chap-core branch with chapkit self-registration feature
- chapkit >= 0.16.7 installed in the test model fixture

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

Environment variables:
- `SERVICEKIT_ORCHESTRATOR_URL`: chap-core's v2 registration endpoint
- `SERVICEKIT_HOST=host.docker.internal`: ensures the service registers with a URL that chap-core's Docker container can resolve (requires chapkit >= 0.16.7)
- `SERVICEKIT_PORT=8090`: the port the service listens on (must match uvicorn `--port`)

The test model uses **seasonal naive prediction**: for each location, it predicts the same month's disease_cases value from the most recent year in training data.

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

This shows the full flow with a known working model. Use this as a reference to verify the DHIS2 setup works before testing the chapkit model.

1. Navigate to http://localhost:8080/apps/dhis2-chapmodeling-app#/evaluate/new
2. Fill in:
   - **Name**: EWARS Test
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

Same flow as Step 5 but select "Chapkit Test Model":

1. Navigate to http://localhost:8080/apps/dhis2-chapmodeling-app#/evaluate/new
2. Fill in same parameters:
   - **Name**: Chapkit Evaluation Test
   - **Period type**: Monthly
   - **From period**: 2023-01
   - **To period**: 2024-12
   - **Organisation units**: Level "Province" (18 provinces)
3. Click "Select model" > choose "Chapkit Test Model" > Confirm
4. Click "Configure sources" and map same data items:
   - Disease cases -> "Dengue Cases (Any) - Monthly"
   - Population -> "Population by year"
   - Rainfall -> "Precipitation (CHIRPS)"
   - Mean temperature -> "Air temperature (ERA5-Land)"
5. Save, dry run (expect "Valid import: All 18 locations"), start import
6. Monitor job on Jobs page - should complete in under 1 minute

## Step 7: Run a prediction with Chapkit Test Model

1. Navigate to http://localhost:8080/apps/dhis2-chapmodeling-app#/predictions (or click "Predict" in sidebar)
2. Click "New prediction"
3. Fill in:
   - **Name**: Chapkit Prediction Test
   - **Period type**: Monthly
   - **From period**: 2023-01
   - **To period**: 2024-12
   - **Organisation units**: Level "Province" (18 provinces)
4. Click "Select model" > choose "Chapkit Test Model" > Confirm
5. Click "Configure sources" and map same 4 data items as evaluation
6. Save, dry run, start import
7. Monitor on Jobs page - should complete in under 30 seconds

### Verify prediction results

After the job completes, click "More" (three-dot menu) in the Actions column, then "Go to result". This opens the prediction details page with:

- **Chart tab**: Shows actual cases (training period) and predicted values (3 future periods)
- **Location selector**: Left sidebar lists all 18 provinces - click to switch between them
- **Legend**: Actual Cases, Median prediction, 80% prediction interval, 50% prediction interval

**Expected behavior for seasonal naive**: predictions should match the same month from the most recent training year. For example, if training data ends at 2024-12 and the model predicts 3 periods ahead:
- 2025-01 prediction should equal 2024-01 actual
- 2025-02 prediction should equal 2024-02 actual
- 2025-03 prediction should equal 2024-03 actual

Since the model only outputs one sample (`sample_0`), the prediction intervals collapse to a single line overlapping the median.

You can also verify via API:
```bash
# List predictions
curl -s http://localhost:8000/v1/crud/predictions | jq '.[].name'

# Get prediction details (replace 1 with actual ID)
curl -s http://localhost:8000/v1/crud/predictions/1 | jq '{name, nPeriods, orgUnits: (.orgUnits | length)}'
```

## Step 8: Test multiple configured models (variant)

1. Create a variant configured model via API:
   ```bash
   # Get the chapkit template ID
   TEMPLATE_ID=$(curl -s http://localhost:8000/v1/crud/model-templates | \
     python3 -c "import json,sys; [print(t['id']) for t in json.load(sys.stdin) if t['name']=='chapkit-test-model']")

   # Create variant (uses_chapkit inherited from template)
   curl -X POST http://localhost:8000/v1/crud/configured-models \
     -H "Content-Type: application/json" \
     -d "{\"name\": \"variant-a\", \"model_template_id\": $TEMPLATE_ID, \"user_option_values\": {}}"
   ```
2. Hard refresh the modeling app (Cmd+Shift+R)
3. Navigate to Evaluate > New evaluation
4. "Chapkit Test Model [Variant-a]" should appear as a separate selectable model
5. Select it, map same covariates, dry run, start import
6. Verify job completes successfully

## Known Issues

### Browser caching
After registering a new chapkit service or creating new configured models, the modeling app may not show them immediately. Hard refresh (Cmd+Shift+R) resolves this.

### DHIS2 route configuration
The route URL must use `host.docker.internal` when DHIS2 runs in Docker and chap-core is on the host or a different Docker network. The default `http://chap-core:8000/**` only works when both are on the same Docker compose network. The route configuration is lost on Docker restart and must be reconfigured each time.

### SERVICEKIT_HOST env var
The `SERVICEKIT_HOST` env var was ignored in older versions. Fixed in servicekit 0.8.2 / chapkit 0.16.7. With the fix, `SERVICEKIT_HOST=host.docker.internal` correctly sets the registration URL so chap-core's Docker container can reach the chapkit service.

### Data item mapping
The modeling app maps model covariates to DHIS2 data elements/indicators. The covariate names in `required_covariates` must match what the app expects:
- `disease_cases` (target) -> "Dengue Cases (Any) - Monthly"
- `population` -> "Population by year"
- `rainfall` -> "Precipitation (CHIRPS)"
- `mean_temperature` -> "Air temperature (ERA5-Land)"

### Multiple configured models per chapkit template
Creating additional configured models from a chapkit template works end-to-end:

```bash
# Create a variant via API (uses_chapkit inherited from template automatically)
curl -X POST http://localhost:8000/v1/crud/configured-models \
  -H "Content-Type: application/json" \
  -d '{"name": "variant-a", "model_template_id": <template-id>, "user_option_values": {}}'
```

- `uses_chapkit` is inherited from the parent template
- Both default and variant models appear in the model selection dialog (after hard refresh)
- Both can run evaluations successfully
- The variant shows as "Chapkit Test Model [Variant-a]" in the UI

### Prediction intervals
The test model outputs a single sample (`sample_0`), so prediction intervals collapse to a single point. A real model would output multiple samples (e.g. `sample_0` through `sample_99`) for meaningful intervals.

## Verified Results

All flows completed successfully through the modeling app:
- Evaluation with EWARS (reference model): SUCCESS (~5-6 min)
- Evaluation with default chapkit model: SUCCESS (<1 min)
- Evaluation with variant-a configured model: SUCCESS (<1 min)
- Prediction with chapkit model: SUCCESS (<30 sec)
- Prediction results show correct seasonal naive values per location and month

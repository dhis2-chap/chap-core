# Backtest Plot System

Guide to the backtest visualization plugin system — how existing plots work, how to add new ones, and how to test them.

## Architecture

Backtest plots are registered via a decorator-based plugin system. Each plot is a single Python class that receives flat pandas DataFrames and returns an Altair chart. The chart is serialized to Vega-Lite JSON and served via the REST API for the modeling app to render.

```
BackTest (DB row)
  → Evaluation (assessment abstraction)
    → Flat DataFrames (observations, forecasts, historical_observations)
      → Plot class .plot() method
        → Altair chart
          → Vega-Lite JSON (served via API)
            → Modeling app renders in browser
```

## Key files

| File | Purpose |
|---|---|
| `chap_core/assessment/backtest_plots/__init__.py` | Base class, decorator, registry |
| `chap_core/assessment/backtest_plots/*.py` | Individual plot implementations |
| `chap_core/rest_api/v1/routers/visualization.py` | API endpoints for serving plots |
| `chap_core/assessment/flat_representations.py` | Flat DataFrame schemas |

## Data schemas

Plots receive three flat DataFrames (not raw BackTest objects):

**`observations`** — ground truth for the evaluation window:
```
location       | time_period | disease_cases
---------------+-------------+--------------
FRmrFTE63D0    | 2024-01     | 42.0
FRmrFTE63D0    | 2024-02     | 67.0
```

**`forecasts`** — model predictions with samples:
```
location       | time_period | horizon_distance | sample | forecast
---------------+-------------+------------------+--------+---------
FRmrFTE63D0    | 2024-01     | 1                | 0      | 45.0
FRmrFTE63D0    | 2024-01     | 1                | 1      | 38.0
...            | ...         | ...              | 999    | 51.0
```

**`historical_observations`** (optional, set `needs_historical=True`):
```
location       | time_period | disease_cases
---------------+-------------+--------------
FRmrFTE63D0    | 2023-01     | 31.0
FRmrFTE63D0    | 2023-02     | 28.0
```

Note: covariate data (rainfall, mean_temperature, population) is NOT available in these DataFrames — only `disease_cases`. Plots that need covariate data would need to extend the data flow (see "Limitations" below).

## Existing plots

| Plot ID | Class | File | Description |
|---|---|---|---|
| `evaluation_plot` | `EvaluationPlot` | `evaluation_plot.py` | Time series with uncertainty bands (q10/q25/q50/q75/q90) |
| `horizon_location_grid` | `HorizonLocationGridPlot` | `horizon_location_grid.py` | Grid of locations × horizons with forecast intervals + metrics |
| `metrics_dashboard` | `MetricsDashboardPlot` | `metrics_dashboard.py` | Multi-metric overview by horizon and time |
| `predicted_vs_actual` | `PredictedVsActualPlot` | `predicted_vs_actual_plot.py` | Scatter plot (median predicted vs actual, log1p space, faceted by horizon) |
| `ratio_of_samples_above_truth` | `SampleBiasPlot` | `sample_bias_plot.py` | Forecast bias relative to observations |

## How to add a new plot

### 1. Create the file

Add a new file under `chap_core/assessment/backtest_plots/`, e.g. `my_plot.py`.

### 2. Implement the class

```console
import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot


@backtest_plot(
    plot_id="my_custom_plot",
    name="My Custom Plot",
    description="Shows something useful about the backtest.",
    needs_historical=False,  # set True if you need historical_observations
)
class MyCustomPlot(BacktestPlotBase):
    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
        covariates: pd.DataFrame | None = None,
    ) -> ChartType:
        # Your Altair chart logic here
        chart = alt.Chart(observations).mark_point().encode(
            x="time_period:N",
            y="disease_cases:Q",
        )
        return chart
```

### 3. Register the import

Add an import to `chap_core/assessment/backtest_plots/__init__.py` at the bottom (in the import block that loads all plot modules):

```console
from chap_core.assessment.backtest_plots import my_plot  # noqa: F401
```

### 4. Verify registration

```console
# Check the plot appears in the listing
curl -s http://localhost:8000/v1/visualization/backtest-plots/ | python3 -m json.tool
```

Your plot should appear in the list with the `id`, `displayName`, and `description` you specified.

### 5. Render it

```console
# Render against an existing backtest (replace 1 with your backtest ID)
curl -s http://localhost:8000/v1/visualization/backtest-plots/my_custom_plot/1 | python3 -m json.tool
```

This returns a Vega-Lite JSON spec. Paste it into https://vega.github.io/editor/ to preview.

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /v1/visualization/backtest-plots/` | List all registered backtest plots |
| `GET /v1/visualization/backtest-plots/{plot_id}/{backtest_id}` | Render a specific plot for a backtest |

Note: there is also a separate `metric-plots` system for per-metric aggregation charts (`metric_by_horizon`, `metric_map`). These are different from backtest-plots — see `visualization.py:metric_plots_registry`.

## Testing a new plot

### Prerequisites

Start the chap-core stack with a completed backtest:

```console
cd chap-sdk/chap-core
make force-restart     # fresh DB, builds from source
# Wait for stack to be healthy
make chap-version
```

Then run an evaluation (via the modeling app or API):

```console
# Via API
curl -sSL -X POST http://localhost:8000/v1/crud/backtests/ \
  -H "Content-Type: application/json" \
  -d '{"modelId": "ewars_template", "datasetId": 1, "name": "plot test"}'
# Wait for job to complete
```

### Verify the plot listing

```console
curl -s http://localhost:8000/v1/visualization/backtest-plots/ | \
  python3 -c "import json,sys; [print(p['id'], '-', p['displayName']) for p in json.load(sys.stdin)]"
```

Expected output includes your new plot ID.

### Render and preview

```console
# Get the backtest ID
BACKTEST_ID=$(curl -s http://localhost:8000/v1/crud/backtests | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['id'])")

# Render the plot
curl -s "http://localhost:8000/v1/visualization/backtest-plots/my_custom_plot/$BACKTEST_ID" > /tmp/plot.json

# Open in Vega editor (copy-paste the JSON)
echo "Paste contents of /tmp/plot.json into https://vega.github.io/editor/"
cat /tmp/plot.json | python3 -m json.tool | head -50
```

### View in the modeling app

Prerequisites (after any DHIS2 restart):

1. **Login**: `http://localhost:8080` with `admin` / `district`
2. **Route config**: Navigate to `http://localhost:8080/apps/dhis2-chapmodeling-app#/settings`, edit the route URL to `http://host.docker.internal:8000/**`, save. Also click "Increase to 30 seconds" if prompted.
3. **Hard refresh** (Cmd+Shift+R) if the model list seems stale.

Then:

1. Open `http://localhost:8080/apps/dhis2-chapmodeling-app#/jobs`
2. Find your completed evaluation, click Actions → "Go to result"
3. The result view renders all registered backtest plots — your new plot should appear

### Playwright testing (automated)

For automated UI testing with the playwright MCP server, key patterns:

- **Login**: the DHIS2 app uses an in-page dialog (`dialog[role]` with Username/Password inputs), not a redirect. Fill inputs via `browser_fill_form` on the textbox refs, then click "Sign in".
- **Route update**: navigate to `#/settings`, click the edit pencil next to "Route configuration", fill the URL input with `http://host.docker.internal:8000/**`, click Save.
- **Iframe access**: the modeling app runs inside an iframe. All DOM queries must go through `document.querySelector('iframe').contentDocument` via `browser_evaluate`.
- **`browser_wait_for` doesn't pierce iframes**: use `browser_evaluate` to poll the iframe DOM directly. Dry runs complete in 1-2 seconds.
- **`data-test` attributes use hyphens**: e.g. `feature-mapping-mean-temperature-trigger` (not underscores).
- **Data element search**: the mapping dropdowns require typing into the search placeholder ("Search for indicators, data elements, or program indicators") before the target option is visible. Use the native setter trick on the input value + `input` event.
- **Hard refresh**: programmatic `location.reload(true)` inside the iframe does NOT reliably bust the modeling app's react-query cache. The user must do Cmd+Shift+R manually if the model list is stale.

### Run linting and tests

```console
make lint
uv run pytest tests/ -q
```

## FE integration — Evaluation Details page

The Vega-based backtest plots render in the **Evaluation details** page at `#/evaluate/{evaluationId}`. This page is accessed from:

1. `#/evaluate` → click on an evaluation row → opens `#/evaluate/{id}`
2. Jobs page → Actions → "Go to result" → redirects to `#/evaluate/compare?baseEvaluation={id}` (the Compare page does NOT show these plots; navigate manually to `#/evaluate/{id}`)

### Enabling the plots (experimental features)

The plots are behind experimental feature toggles. To enable:

1. Navigate to `#/settings/experimental`
2. Enable "Enable experimental features" (master toggle)
3. Enable **"Evaluation plots"** — shows the Vega-based backtest plots (evaluation_plot, predicted_vs_actual, sample_bias_plot, etc.)
4. Enable **"Metric plots"** — shows per-metric visualization plots (CRPS by horizon, RMSE map, etc.)

### What renders where

On the **Evaluation details** page (`#/evaluate/{id}`), below the main Highcharts time-series chart:

- **Evaluation plots** section (collapsible widget) — dropdown with all registered `backtest_plots`:
  - Evaluation Plot
  - Forecast Grid (Locations x Horizons)
  - Overview of various metrics by horizon/time
  - Predicted vs Actual (scatter plot)
  - Regional RMSE distribution
  - Sample Bias Plot

- **Metric plot** section (collapsible widget) — two dropdowns:
  - Visualization type: Horizon Plot, Map
  - Metric: CRPS, MAPE, RMSE, Coverage, Winkler, etc. (all 15+ metrics)

**New plots added via `@backtest_plot()` will automatically appear in the "Evaluation plots" dropdown** — no FE changes needed.

The **Compare evaluations** page (`#/evaluate/compare`) and **Prediction details** page (`#/predictions/{id}`) use their own Highcharts-based charts and do NOT render the Vega-based backtest plots.

### FE code references

| File | Purpose |
|---|---|
| `apps/modeling-app/src/components/PageContent/EvaluationDetails/EvaluationDetails.component.tsx` | Main page, conditionally renders `CustomEvaluationPlotsWidget` and `MetricPlotWidget` based on experimental feature toggles |
| `apps/modeling-app/src/components/PageContent/EvaluationDetails/CustomEvaluationPlots/` | Fetches plot types from `/v1/visualization/backtest-plots/`, renders selected plot via Vega |
| `apps/modeling-app/src/components/PageContent/EvaluationDetails/MetricPlot/` | Fetches metric types from `/v1/visualization/metrics/`, renders via Vega |
| `apps/modeling-app/src/features/settings/Experimental/` | Experimental feature toggle management |

## CLIM-538 implementation notes (Uganda nutrition plots)

### Plot 1: Predicted vs Actual (linear) — `predicted_vs_actual_linear`

Scatter plot of predicted (median) vs observed values with OLS regression line, colored by location.

Differences from the original reference to revisit after feedback:

- **Risk-level coloring**: The reference colors by Risk Level (Normal/Alert/Alarm/Emergency thresholds). Our version colors by location since chap-core doesn't define outbreak thresholds per model.
- **Summary metrics header**: The reference shows MAE, RMSE, CV Folds above the chart. These are available in `aggregateMetrics` but not rendered as chart annotations.

### Plot 2: Covariate Importance Radar — `covariate_importance`

Radar/spider chart showing signed Spearman rank correlations between each covariate and disease cases. Blue spokes = positive correlation, red = negative. Uses a raw Vega spec since Altair/Vega-Lite does not support radial layouts.

This plot requires covariate data and is only available via `create_plot_from_backtest()` (not from evaluation files). The `needs_covariates=True` flag on the decorator triggers covariate extraction from the BackTest's dataset.

## Limitations

- **Covariate data is only available from BackTest objects.** Plots with `needs_covariates=True` (e.g. `covariate_importance`) cannot be generated from evaluation files — `create_plot_from_evaluation()` will raise `ValueError`. The `_extract_covariates()` helper in `__init__.py` handles extraction from `BackTest.dataset.observations`.

- **Altair and raw Vega.** Most plots return Altair chart objects serialized via `chart.to_dict(format="vega")`. Plots needing chart types not supported by Vega-Lite (e.g. radar charts) can return raw Vega spec dicts instead. The API endpoint and CLI handle both types.

- **No interactivity beyond Vega.** The frontend renders the Vega spec as-is. Tooltips work; custom JavaScript callbacks do not.

## Reference: pattern from `predicted_vs_actual_plot.py`

This is the most recent and cleanest example to copy from:

1. Compute median forecasts from the samples: `forecasts.groupby([...]).agg(median_forecast=("forecast", "median"))`
2. Merge with observations: `median_forecasts.merge(observations, on=[...])`
3. Build the Altair chart with encoding, layering, faceting
4. Return the chart

See `chap_core/assessment/backtest_plots/predicted_vs_actual_plot.py` for the full implementation.

# Creating Custom Backtest Plots

This guide explains how to create custom visualizations for CHAP backtest results using the backtest plot plugin system.

## Overview

Backtest plots are visualizations that display forecast evaluation results. CHAP provides a plugin system that allows you to create custom plots that integrate seamlessly with the evaluation workflow.

Each backtest plot:
- Receives flat pandas DataFrames containing observations and forecasts
- Returns an Altair chart
- Is automatically registered and discoverable
- **Is automatically available in the CHAP Modeling App** through the REST API once registered

## Data Schemas

Your plot will receive data in standardized DataFrame formats:

### Observations DataFrame

Contains the actual observed values:

| Column | Type | Description |
|--------|------|-------------|
| `location` | str | Location identifier |
| `time_period` | str | Time period (e.g., "2024-01" or "2024W01") |
| `disease_cases` | float | Observed disease cases |

### Forecasts DataFrame

Contains forecast samples:

| Column | Type | Description |
|--------|------|-------------|
| `location` | str | Location identifier |
| `time_period` | str | Time period being forecasted |
| `horizon_distance` | int | How many periods ahead this forecast is |
| `sample` | int | Sample index (for probabilistic forecasts) |
| `forecast` | float | Forecasted value |

### Historical Observations DataFrame (Optional)

If your plot sets `needs_historical=True`, it also receives historical data with the same schema as observations.

### Example Data

Here's example data that demonstrates the expected format:

```python
import pandas as pd

observations = pd.DataFrame({
    "location": ["loc1", "loc1", "loc2", "loc2"],
    "time_period": ["2023-W01", "2023-W02", "2023-W01", "2023-W02"],
    "disease_cases": [100.0, 120.0, 80.0, 95.0],
})

forecasts = pd.DataFrame({
    "location": ["loc1", "loc1", "loc2", "loc2"] * 2,
    "time_period": ["2023-W01", "2023-W02", "2023-W01", "2023-W02"] * 2,
    "horizon_distance": [1, 1, 1, 1, 2, 2, 2, 2],
    "sample": [0] * 8,
    "forecast": [95.0, 115.0, 78.0, 90.0, 102.0, 122.0, 85.0, 97.0],
})
```

The observations DataFrame looks like:

```text
  location time_period  disease_cases
0     loc1    2023-W01          100.0
1     loc1    2023-W02          120.0
2     loc2    2023-W01           80.0
3     loc2    2023-W02           95.0
```

The forecasts DataFrame looks like:

```text
  location time_period  horizon_distance  sample  forecast
0     loc1    2023-W01                 1       0      95.0
1     loc1    2023-W02                 1       0     115.0
2     loc2    2023-W01                 1       0      78.0
3     loc2    2023-W02                 1       0      90.0
4     loc1    2023-W01                 2       0     102.0
5     loc1    2023-W02                 2       0     122.0
6     loc2    2023-W01                 2       0      85.0
7     loc2    2023-W02                 2       0      97.0
```

## Creating a Basic Plot

### Step 1: Import Requirements

```python
from typing import Optional
import pandas as pd
import altair as alt
from chap_core.assessment.backtest_plots import backtest_plot, BacktestPlotBase, ChartType
```

### Step 2: Define Your Plot Class

Use the `@backtest_plot` decorator to register your plot, then inherit from `BacktestPlotBase` and implement the `plot()` method:

```python
from typing import Optional
import pandas as pd
from chap_core.assessment.backtest_plots import backtest_plot, BacktestPlotBase, ChartType

@backtest_plot(
    plot_id="my_custom_plot",              # Unique identifier (used in APIs)
    name="My Custom Plot",             # Human-readable display name
    description="Shows forecast accuracy by location.",
)
class MyCustomPlot(BacktestPlotBase):
    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: Optional[pd.DataFrame] = None,
    ) -> ChartType:
        # Your visualization logic here
        chart = None
        return chart
```

## Complete Example: Forecast Error by Location

Here's a complete working example that shows mean absolute error by location:

```python
from typing import Optional
import pandas as pd
import altair as alt
from chap_core.assessment.backtest_plots import backtest_plot, BacktestPlotBase, ChartType


@backtest_plot(
    plot_id="error_by_location",
    name="Error by Location",
    description="Shows mean absolute forecast error for each location.",
)
class ErrorByLocationPlot(BacktestPlotBase):
    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: Optional[pd.DataFrame] = None,
    ) -> ChartType:
        # Compute median forecast for each location/time_period
        median_forecasts = (
            forecasts.groupby(["location", "time_period"])["forecast"]
            .median()
            .reset_index()
        )

        # Merge with observations
        merged = median_forecasts.merge(
            observations, on=["location", "time_period"]
        )

        # Calculate absolute error
        merged["abs_error"] = abs(merged["forecast"] - merged["disease_cases"])

        # Aggregate by location
        error_by_loc = (
            merged.groupby("location")["abs_error"]
            .mean()
            .reset_index()
            .rename(columns={"abs_error": "mean_abs_error"})
        )

        # Create bar chart
        chart = (
            alt.Chart(error_by_loc)
            .mark_bar()
            .encode(
                x=alt.X("location:N", title="Location"),
                y=alt.Y("mean_abs_error:Q", title="Mean Absolute Error"),
                tooltip=["location", "mean_abs_error"],
            )
            .properties(
                width=400,
                height=300,
                title="Mean Absolute Error by Location",
            )
        )

        return chart
```

## Registration and Discovery

### How Registration Works

The `@backtest_plot` decorator registers your plot class in a global registry when the module is imported. The decorator:
1. Validates that your class inherits from `BacktestPlotBase`
2. Assigns the metadata (`id`, `name`, `description`, `needs_historical`) to the class
3. Adds the class to the registry under its `id`

### Making CHAP Discover Your Plot

For CHAP to discover your plot at startup, you need to import your module in the `_discover_plots()` function in `chap_core/assessment/backtest_plots/__init__.py`:

```console
def _discover_plots():
    """Import all plot modules to trigger decorator registration."""
    from chap_core.assessment.backtest_plots import (
        metrics_dashboard,
        sample_bias_plot,
        evaluation_plot,
        my_custom_plot,  # Add your module here
    )
```

### Where to Place Your Plot File

Create your plot file in `chap_core/assessment/backtest_plots/`. For example:
- `chap_core/assessment/backtest_plots/error_by_location_plot.py`

## Testing Your Plot

### Using create_plot_from_evaluation

You can test your plot with an `Evaluation` object:

```python
from chap_core.assessment.backtest_plots import create_plot_from_evaluation
from chap_core.assessment.evaluation import Evaluation

# Load an evaluation from the example file
evaluation = Evaluation.from_file("example_data/example_evaluation.nc")

# Create a plot (using a built-in plot type)
chart = create_plot_from_evaluation("ratio_of_samples_above_truth", evaluation)

# Save to HTML for inspection (uncomment to save)
# chart.save("my_plot.html")
```

### Unit Testing

Write a test for your plot in `tests/evaluation/test_backtest_plot.py`:

```console
def test_my_custom_plot_directly(flat_observations, flat_forecasts, default_transformer):
    """Test my custom plot with flat data."""
    from chap_core.assessment.backtest_plots.my_custom_plot import MyCustomPlot
    import pandas as pd

    plot = MyCustomPlot()
    chart = plot.plot(pd.DataFrame(flat_observations), pd.DataFrame(flat_forecasts))
    assert chart is not None
```

The `flat_observations` and `flat_forecasts` fixtures are defined in `tests/evaluation/conftest.py`.

## Using Your Plot

### Generating Plots with the CLI

Once your plot is registered, you can generate it from the command line using evaluation data from `chap eval`:

```console
# First, run an evaluation to generate the .nc file
chap eval my_model data.csv evaluation.nc

# Then generate your custom plot
chap plot-backtest evaluation.nc my_plot.html --plot-type my_custom_plot
```

The `plot-backtest` command supports multiple output formats:
- `.html` - Interactive Vega-Lite chart (recommended for exploration)
- `.png` - Static image
- `.svg` - Vector graphics
- `.pdf` - PDF document
- `.json` - Raw Vega specification

To see all available plot types:

```console
chap plot-backtest --help
```

### Automatic Integration with CHAP Modeling App

Once your plot is registered in CHAP, it becomes **automatically available in the CHAP Modeling App** through the REST API. Users of the modeling app will be able to select your plot from the available visualization options when viewing backtest results.

The REST API exposes your plot through these endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /visualization/backtest-plots/` | Lists all available plot types (including yours) |
| `GET /visualization/backtest-plots/{plot_id}/{backtest_id}` | Generates your plot for a specific backtest |

This means:
- No additional frontend work is required
- Your plot appears alongside built-in plots in the UI
- Users can access your visualization without any code changes

## Advanced Topics

### Using Historical Observations

If your plot needs historical context (observations from before the test period), set `needs_historical=True`:

```python
from typing import Optional
import pandas as pd
from chap_core.assessment.backtest_plots import backtest_plot, BacktestPlotBase, ChartType

@backtest_plot(
    plot_id="trend_plot",
    name="Trend Plot",
    description="Shows forecasts with historical trend context.",
    needs_historical=True,  # Request historical data
)
class TrendPlot(BacktestPlotBase):
    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: Optional[pd.DataFrame] = None,
    ) -> ChartType:
        # historical_observations will now be populated
        if historical_observations is not None:
            # Use historical data for context
            pass
        return None
```

### Returning Different Chart Types

The `plot()` method can return any of these Altair chart types:
- `alt.Chart` - Basic chart
- `alt.LayerChart` - Layered charts (multiple marks on same axes)
- `alt.VConcatChart` - Vertically concatenated charts
- `alt.HConcatChart` - Horizontally concatenated charts
- `alt.FacetChart` - Faceted charts (small multiples)

Example with vertical concatenation:

```console
def plot(self, observations, forecasts, historical_observations=None):
    chart1 = alt.Chart(...).mark_bar().encode(...)
    chart2 = alt.Chart(...).mark_line().encode(...)
    return alt.vconcat(chart1, chart2)
```

### Using FlatObserved and FlatForecasts Wrappers

For convenience, you can wrap the DataFrames in typed wrappers:

```console
from chap_core.assessment.flat_representations import FlatObserved, FlatForecasts

def plot(self, observations, forecasts, historical_observations=None):
    flat_obs = FlatObserved(observations)
    flat_fcst = FlatForecasts(forecasts)
    # These provide the same interface as DataFrames
    ...
```

## Reference

### Existing Implementations

Study these existing implementations as examples:

| File | Description |
|------|-------------|
| `sample_bias_plot.py` | Simple plot showing forecast bias |
| `metrics_dashboard.py` | Dashboard with multiple metrics |
| `evaluation_plot.py` | Complex plot with historical context |

### API Reference

#### `@backtest_plot` Decorator

```console
@backtest_plot(
    plot_id: str,                    # Required: Unique identifier
    name: str,                  # Required: Display name
    description: str = "",      # Optional: Description
    needs_historical: bool = False,  # Optional: Request historical data
)
```

#### `BacktestPlotBase` Class

Abstract base class with one required method:

```console
def plot(
    self,
    observations: pd.DataFrame,
    forecasts: pd.DataFrame,
    historical_observations: Optional[pd.DataFrame] = None,
) -> ChartType:
    """Generate the visualization."""
    pass
```

#### Helper Functions

```python
# Get all registered plots
from chap_core.assessment.backtest_plots import get_backtest_plots_registry
registry = get_backtest_plots_registry()

# Get a specific plot class
from chap_core.assessment.backtest_plots import get_backtest_plot
plot_cls = get_backtest_plot("ratio_of_samples_above_truth")

# List all plots with metadata
from chap_core.assessment.backtest_plots import list_backtest_plots
plots = list_backtest_plots()
```

# Creating Custom Metric Plots

This guide explains how to create custom metric visualizations for Chap backtest results using the metric plot plugin system.

## Overview

Metric plots visualize pre-computed metric values (e.g., RMSE, CRPS) across locations, time periods, and forecast horizons. Chap provides a plugin system that lets you create custom plots that integrate seamlessly with the evaluation workflow.

Each metric plot:
- Receives a pre-computed metric DataFrame (computed from raw backtest data using a chosen `Metric`)
- Returns an Altair chart
- Is automatically registered and discoverable
- **Is automatically available in the Chap Modeling App** through the REST API once registered

The user chooses which metric to apply (RMSE, CRPS, etc.) when requesting the plot — the plot class focuses only on how to visualize the values, not how to compute them.

## Data Schema

### Metric DataFrame

Your plot receives a single DataFrame with pre-computed metric values:

| Column | Type | Description |
|--------|------|-------------|
| `location` | str | Location identifier |
| `time_period` | str | Time period (e.g., "2024-01" or "2024W01") |
| `horizon_distance` | int | How many periods ahead this forecast is |
| `metric` | float | Computed metric value (e.g., RMSE or CRPS per row) |

Each row represents one (location, time_period, horizon_distance) combination.

### Example Data

```python
import pandas as pd

metric_data = pd.DataFrame({
    "location": ["loc1", "loc1", "loc2", "loc2"] * 2,
    "time_period": ["2023-W01", "2023-W02", "2023-W01", "2023-W02"] * 2,
    "horizon_distance": [1, 1, 1, 1, 2, 2, 2, 2],
    "metric": [0.12, 0.18, 0.09, 0.14, 0.20, 0.25, 0.15, 0.19],
})
```

The DataFrame looks like:

```text
  location time_period  horizon_distance  metric
0     loc1    2023-W01                 1    0.12
1     loc1    2023-W02                 1    0.18
2     loc2    2023-W01                 1    0.09
3     loc2    2023-W02                 1    0.14
4     loc1    2023-W01                 2    0.20
5     loc1    2023-W02                 2    0.25
6     loc2    2023-W01                 2    0.15
7     loc2    2023-W02                 2    0.19
```

## Creating a Plot

### Step 1: Import Requirements

```python
import altair as alt
from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot
```

### Step 2: Define Your Plot Class

Use the `@metric_plot` decorator to register your plot, then inherit from `MetricPlotBase` and implement `plot_from_df`:

```python
import altair as alt
from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot


@metric_plot(
    plot_id="my_metric_plot",
    name="My Metric Plot",
    description="Shows mean metric value per forecast horizon.",
)
class MyMetricPlot(MetricPlotBase):
    def plot_from_df(self, title: str = "My Metric Plot") -> alt.Chart:
        df = self._metric_data  # pre-computed metric DataFrame
        adf = df.groupby("horizon_distance").agg({"metric": "mean"}).reset_index()
        return (
            alt.Chart(adf)
            .mark_bar()
            .encode(
                x=alt.X("horizon_distance:O", title="Horizon (periods ahead)"),
                y=alt.Y("metric:Q", title="Mean Metric Value"),
                tooltip=["horizon_distance", "metric"],
            )
            .properties(width=400, height=300, title=title)
        )
```

## Complete Example: Mean Metric by Location Over Time

Here is a full working example that plots mean metric per location as a line chart:

```python
import altair as alt
from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot


@metric_plot(
    plot_id="metric_by_location_time",
    name="Metric by Location over Time",
    description="Line chart showing mean metric per location across time periods.",
)
class MetricByLocationTimePlot(MetricPlotBase):
    def plot_from_df(self, title: str = "Metric by Location over Time") -> alt.Chart:
        df = self._metric_data
        adf = df.groupby(["time_period", "location"]).agg({"metric": "mean"}).reset_index()
        return (
            alt.Chart(adf)
            .mark_line(point=True)
            .encode(
                x=alt.X("time_period:O", title="Time Period"),
                y=alt.Y("metric:Q", title="Mean Metric"),
                color=alt.Color("location:N", title="Location"),
                tooltip=["time_period", "location", "metric"],
            )
            .properties(width=500, height=300, title=title)
        )
```

Test it directly with example data:

```python
import pandas as pd

example_data = pd.DataFrame({
    "location": ["loc1", "loc2", "loc1", "loc2"],
    "time_period": ["2023-W01", "2023-W01", "2023-W02", "2023-W02"],
    "horizon_distance": [1, 1, 1, 1],
    "metric": [0.12, 0.09, 0.18, 0.14],
})

chart = MetricByLocationTimePlot(example_data).plot_from_df()
assert chart is not None
```

## Registration and Discovery

### How Registration Works

The `@metric_plot` decorator registers your plot class in a global registry when the module is imported. The decorator:

1. Validates that your class inherits from `MetricPlotBase`
2. Assigns `id`, `name`, and `description` to the class as attributes
3. Adds the class to the registry under its `id`

### Making Chap Discover Your Plot

Create your file in `chap_core/assessment/metric_plots/` and add an import to `_discover_plots()` in `chap_core/assessment/metric_plots/__init__.py`:

```console
def _discover_plots() -> None:
    from chap_core.assessment.metric_plots import (
        horizon_mean,
        metric_map,
        regional_distribution,
        my_new_plot,  # add your module here
    )
```

### Where to Place Your File

```text
chap_core/assessment/metric_plots/
    __init__.py          # plugin system — edit _discover_plots() here
    horizon_mean.py      # example: existing registered plot
    my_new_plot.py       # your new plot
```

## Testing Your Plot

Use fixtures from `tests/evaluation/conftest.py` to get a realistic backtest object:

```python
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics.mae import MAEMetric
from chap_core.plotting.evaluation_plot import make_plot_from_evaluation_object
from chap_core.assessment.metric_plots.horizon_mean import MetricByHorizonV2Mean
```

In a pytest test:

```console
def test_my_metric_plot(backtest):
    evaluation = Evaluation.from_backtest(backtest)
    flat_data = evaluation.to_flat()
    metric_data = MAEMetric().get_detailed_metric(flat_data.observations, flat_data.forecasts)

    chart = MyMetricPlot(metric_data).plot_from_df()

    assert chart is not None
```

The `backtest` fixture is available from `tests/evaluation/conftest.py`.

## Automatic REST Exposure

Once your plot is registered, it is automatically available through two REST endpoints:

**List available metric plots:**

```text
GET /visualization/metric-plots/{backtest_id}
```

Returns a list of all registered plots with their `id`, `display_name`, and `description`.

**Generate a metric plot:**

```text
GET /visualization/metric-plots/{plot_id}/{backtest_id}/{metric_id}
```

For example:

```text
GET /visualization/metric-plots/metric_by_location_time/42/crps
```

This computes the CRPS metric from backtest 42 and renders it with your plot class. The response is a Vega chart specification (JSON) ready for the frontend to render.

Available `metric_id` values correspond to registered metrics such as `rmse`, `crps`, `mae`.

## Comparison with BacktestPlot

Metric plots and backtest plots serve different purposes:

| | MetricPlot | BacktestPlot |
|---|---|---|
| **Input** | Pre-computed metric DataFrame | Raw observations & forecasts DataFrames |
| **Metric choice** | Delegated to the caller (chosen at request time) | Baked into the plot class |
| **Typical use** | Compare a single metric across dimensions | Richer visualizations combining observations and forecasts |
| **Decorator** | `@metric_plot(plot_id, name, description)` | `@backtest_plot(plot_id, name, description, needs_historical)` |
| **Base class** | `MetricPlotBase` | `BacktestPlotBase` |

Use a MetricPlot when the visualization is purely about displaying a computed metric value. Use a BacktestPlot when you need access to the raw forecast samples or observations directly.

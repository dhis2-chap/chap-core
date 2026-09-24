# Creating Custom Metrics

This guide explains how to create custom evaluation metrics for Chap backtest results.

## Overview

Metrics in Chap measure how well a model's forecasts match observed values. The metrics system provides:

- **Single definition**: Each metric is defined once and supports multiple aggregation levels
- **Multi-level aggregation**: Get global values, per-location, per-horizon, or detailed breakdowns
- **Automatic registration**: Metrics are discovered and available throughout Chap
- **Two metric types**: Deterministic (point forecasts) and Probabilistic (all samples)

## Quick Start

Here's a minimal deterministic metric:

```python
from chap_core.assessment.metrics.base import (
    AggregationOp,
    DeterministicMetric,
    MetricSpec,
    OptimizationDirection,
)
from chap_core.assessment.metrics import metric


@metric()
class MyAbsoluteErrorMetric(DeterministicMetric):
    """Computes absolute error between forecast and observation."""

    spec = MetricSpec(
        metric_id="my_absolute_error",
        metric_name="My Absolute Error",
        aggregation_op=AggregationOp.MEAN,
        description="Absolute difference between forecast and observation",
        optimization_direction=OptimizationDirection.MINIMIZE,
    )

    def compute_point_metric(self, forecast: float, observed: float) -> float:
        return abs(forecast - observed)
```

And a minimal probabilistic metric:

```python
import numpy as np
from chap_core.assessment.metrics.base import (
    AggregationOp,
    ProbabilisticMetric,
    MetricSpec,
    OptimizationDirection,
)
from chap_core.assessment.metrics import metric


@metric()
class MySpreadMetric(ProbabilisticMetric):
    """Computes the spread (std dev) of forecast samples."""

    spec = MetricSpec(
        metric_id="my_spread",
        metric_name="My Spread",
        aggregation_op=AggregationOp.MEAN,
        description="Standard deviation of forecast samples",
        optimization_direction=OptimizationDirection.MINIMIZE,
    )

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        return float(np.std(samples))
```

## Data Formats

Your metric receives data in standardized DataFrame formats:

### Observations DataFrame (FlatObserved)

| Column | Type | Description |
|--------|------|-------------|
| `location` | str | Location identifier |
| `time_period` | str | Time period (e.g., "2024-01" or "2024W01") |
| `disease_cases` | float | Observed disease cases |

### Forecasts DataFrame (FlatForecasts)

| Column | Type | Description |
|--------|------|-------------|
| `location` | str | Location identifier |
| `time_period` | str | Time period being forecasted |
| `horizon_distance` | int | How many periods ahead this forecast is |
| `sample` | int | Sample index (for probabilistic forecasts) |
| `forecast` | float | Forecasted value |

### Output Format

Metrics return a DataFrame with dimension columns plus a `metric` column.

## Base Classes

### DeterministicMetric

For metrics comparing point forecasts (median of samples) to observations:

```python
from chap_core.assessment.metrics.base import DeterministicMetric

# DeterministicMetric requires implementing:
# def compute_point_metric(self, forecast: float, observed: float) -> float
```

### ProbabilisticMetric

For metrics that need all forecast samples:

```python
from chap_core.assessment.metrics.base import ProbabilisticMetric

# ProbabilisticMetric requires implementing:
# def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float
```

## MetricSpec Configuration

```python
from chap_core.assessment.metrics.base import AggregationOp, MetricSpec, TargetBehavior

spec = MetricSpec(
    metric_id="unique_id",              # Used in APIs and registry
    metric_name="Display Name",          # Human-readable name
    aggregation_op=AggregationOp.MEAN,   # MEAN, SUM, or ROOT_MEAN_SQUARE
    description="What this metric measures",
    optimization_direction=None,        # MINIMIZE, MAXIMIZE or None
    proper_scoring_rule=False,          # True only for proper scoring rules, which HPO may optimize
    unit=None,                          # Display suffix for the raw score, e.g. "%"
    target=None,                        # Ideal raw value when neither direction is better, e.g. 0.8
    target_behavior=TargetBehavior.CLOSEST,  # CLOSEST or AT_LEAST, only used with a target
)
```

The metric catalogue API returns `unit`, `target` and `target_behavior` alongside
the optimization direction. A metric with `optimization_direction=None` should set
a `target`, and `target_behavior` tells clients how to judge a score against it:
`CLOSEST` means deviating in either direction is worse (ratio above truth, peak
difference), while `AT_LEAST` means higher is better up to the target and flat
above it, so only scores below the target should be flagged as bad (coverage
metrics). Units do not rescale scores: MAPE is already a percentage, while
coverage targets use fractions such as `0.8`.

`optimization_direction` says which way is better when reading a score, not that the
metric is a sound thing to optimize. Set `proper_scoring_rule=True` only when the metric
is a proper scoring rule (or a consistent scoring function for point metrics, such as
MAE for the median and RMSE for the mean), meaning a forecaster cannot improve its
expected score by reporting anything other than its honest forecast. HPO only accepts
proper scoring rules with an explicit direction. The outbreak metrics keep `MAXIMIZE`
so clients colour high scores as good, but HPO rejects them, since sensitivity is
maximised by always alerting and specificity by never alerting. MAPE is rejected for
the same reason: it is minimised by systematically under-forecasting.

## Complete Examples

### Example: RMSE-style Metric

```python
from chap_core.assessment.metrics.base import (
    AggregationOp,
    DeterministicMetric,
    MetricSpec,
    OptimizationDirection,
)
from chap_core.assessment.metrics import metric


@metric()
class SquaredErrorMetric(DeterministicMetric):
    """
    Squared error metric.

    With ROOT_MEAN_SQUARE aggregation, this produces RMSE.
    """

    spec = MetricSpec(
        metric_id="squared_error",
        metric_name="Squared Error",
        aggregation_op=AggregationOp.ROOT_MEAN_SQUARE,
        description="Squared error with RMSE aggregation",
        optimization_direction=OptimizationDirection.MINIMIZE,
    )

    def compute_point_metric(self, forecast: float, observed: float) -> float:
        return abs(forecast - observed)  # Base class squares for ROOT_MEAN_SQUARE
```

### Example: Bias Detection Metric

```python
import numpy as np
from chap_core.assessment.metrics.base import (
    AggregationOp,
    ProbabilisticMetric,
    MetricSpec,
)
from chap_core.assessment.metrics import metric


@metric()
class ForecastBiasMetric(ProbabilisticMetric):
    """
    Measures forecast bias as proportion of samples above truth.

    Returns 0.5 for unbiased forecasts, >0.5 for over-prediction,
    <0.5 for under-prediction.
    """

    spec = MetricSpec(
        metric_id="forecast_bias",
        metric_name="Forecast Bias",
        aggregation_op=AggregationOp.MEAN,
        description="Proportion of samples above observed (0.5 = unbiased)",
        optimization_direction=None,
        target=0.5,
    )

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        return float(np.mean(samples > observed))
```

### Example: Parameterized Metric with Subclasses

```python
import numpy as np
from chap_core.assessment.metrics.base import (
    AggregationOp,
    ProbabilisticMetric,
    MetricSpec,
    TargetBehavior,
)
from chap_core.assessment.metrics import metric


class IntervalCoverageMetric(ProbabilisticMetric):
    """Base class for interval coverage metrics (not registered directly)."""

    low_pct: int
    high_pct: int

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        low, high = np.percentile(samples, [self.low_pct, self.high_pct])
        return 1.0 if (low <= observed <= high) else 0.0


@metric()
class Coverage80Metric(IntervalCoverageMetric):
    """80% prediction interval coverage."""

    spec = MetricSpec(
        metric_id="coverage_80",
        metric_name="80% Coverage",
        aggregation_op=AggregationOp.MEAN,
        description="Proportion within 10th-90th percentile",
        optimization_direction=None,
        target=0.8,
        target_behavior=TargetBehavior.AT_LEAST,
    )
    low_pct = 10
    high_pct = 90
```

## Using Metrics

### Creating Example Data

First, let's create sample data to demonstrate metric computation:

```python
import pandas as pd
import numpy as np
from chap_core.assessment.flat_representations import FlatObserved, FlatForecasts

# Create sample observations: 2 locations, 3 time periods
observations_df = pd.DataFrame({
    "location": ["loc_A", "loc_A", "loc_A", "loc_B", "loc_B", "loc_B"],
    "time_period": ["2024-01", "2024-02", "2024-03", "2024-01", "2024-02", "2024-03"],
    "disease_cases": [100.0, 120.0, 90.0, 200.0, 180.0, 220.0],
})
observations = FlatObserved(observations_df)

# Create sample forecasts: 10 samples per observation, horizon 1 and 2
forecast_rows = []
np.random.seed(42)
for loc in ["loc_A", "loc_B"]:
    base = 100 if loc == "loc_A" else 200
    for period in ["2024-01", "2024-02", "2024-03"]:
        for horizon in [1, 2]:
            for sample_id in range(10):
                forecast_rows.append({
                    "location": loc,
                    "time_period": period,
                    "horizon_distance": horizon,
                    "sample": sample_id,
                    "forecast": base + np.random.normal(0, 15),
                })
forecasts_df = pd.DataFrame(forecast_rows)
forecasts = FlatForecasts(forecasts_df)

print(f"Observations shape: {observations_df.shape}")
print(f"Forecasts shape: {forecasts_df.shape}")
```

### Computing Metrics at Different Aggregation Levels

```python
from chap_core.assessment.metrics import get_metric
from chap_core.assessment.flat_representations import DataDimension

# Get the MAE metric
mae = get_metric("mae")()

# Global aggregate: single value across all data
global_result = mae.get_global_metric(observations, forecasts)
print("Global MAE:")
print(global_result)
print()

# Detailed: one value per (location, time_period, horizon_distance)
detailed_result = mae.get_detailed_metric(observations, forecasts)
print("Detailed MAE (first 6 rows):")
print(detailed_result.head(6))
print()

# Per location only
per_location = mae.get_metric(observations, forecasts, dimensions=(DataDimension.location,))
print("MAE per location:")
print(per_location)
print()

# Per horizon only
per_horizon = mae.get_metric(observations, forecasts, dimensions=(DataDimension.horizon_distance,))
print("MAE per horizon:")
print(per_horizon)
```

### Getting Metrics from the Registry

```python
from chap_core.assessment.metrics import get_metric, list_metrics

# Get a specific metric by ID
MAEClass = get_metric("mae")
mae_metric = MAEClass()
print(f"Metric: {mae_metric.get_name()} ({mae_metric.get_id()})")
print(f"Description: {mae_metric.get_description()}")
print()

# List all available metrics
print("Available metrics:")
for info in list_metrics():
    print(f"  {info['id']}: {info['name']}")
```

## Registration and Discovery

### The @metric() Decorator

The decorator registers your metric class when the module is imported:

```python
from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import DeterministicMetric, MetricSpec, AggregationOp, OptimizationDirection


@metric()  # This registers the class in the global registry
class RegisteredMetric(DeterministicMetric):
    spec = MetricSpec(
        metric_id="registered_example",
        metric_name="Registered Example",
        aggregation_op=AggregationOp.MEAN,
        description="Example of a registered metric",
        optimization_direction=OptimizationDirection.MINIMIZE,
    )

    def compute_point_metric(self, forecast: float, observed: float) -> float:
        return abs(forecast - observed)
```

### File Location

Place your metric file in `chap_core/assessment/metrics/` and add an import to `_discover_metrics()` in `chap_core/assessment/metrics/__init__.py`.

## Understanding Aggregation

### AggregationOp Options

| Operation | Description | Use Case |
|-----------|-------------|----------|
| `MEAN` | Average of values | MAE, coverage metrics |
| `SUM` | Sum of values | Count-based metrics |
| `ROOT_MEAN_SQUARE` | sqrt(mean(x^2)) | RMSE |

### DataDimension Options

| Dimension | Description |
|-----------|-------------|
| `location` | Geographic location |
| `time_period` | Time period of the forecast |
| `horizon_distance` | How far ahead the forecast is |

## Testing Your Metric

Use existing metrics as a pattern for testing:

```python
from chap_core.assessment.metrics import get_metric

# Verify your metric is registered
metric_cls = get_metric("mae")
assert metric_cls is not None

# Instantiate and check properties
metric = metric_cls()
assert metric.get_id() == "mae"
assert metric.get_name() == "MAE"
```

## Reference

### Existing Implementations

Study these files in `chap_core/assessment/metrics/`:

| File | Type | Description |
|------|------|-------------|
| `mae.py` | Deterministic | Simple absolute error |
| `rmse.py` | Deterministic | Uses ROOT_MEAN_SQUARE aggregation |
| `crps.py` | Probabilistic | Uses all samples |
| `percentile_coverage.py` | Probabilistic | Parameterized with subclasses |
| `above_truth.py` | Probabilistic | Bias detection |

### API Summary

```python
from chap_core.assessment.metrics import (
    metric,              # Decorator to register metrics
    get_metric,          # Get metric class by ID
    get_metrics_registry,  # Get all registered metrics
    list_metrics,        # List metrics with metadata
)
from chap_core.assessment.metrics.base import (
    Metric,              # Base class (abstract)
    DeterministicMetric, # For point forecast comparison
    ProbabilisticMetric, # For sample-based metrics
    MetricSpec,          # Configuration dataclass
    AggregationOp,       # MEAN, SUM, ROOT_MEAN_SQUARE
    OptimizationDirection,  # For hpo objective
)
from chap_core.assessment.flat_representations import DataDimension
```

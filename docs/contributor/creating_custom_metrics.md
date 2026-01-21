# Creating Custom Metrics

This guide explains how to create custom evaluation metrics for CHAP backtest results.

## Overview

Metrics in CHAP measure how well a model's forecasts match observed values. The metrics system provides:

- **Single definition**: Each metric is defined once and supports multiple aggregation levels
- **Multi-level aggregation**: Get global values, per-location, per-horizon, or detailed breakdowns
- **Automatic registration**: Metrics are discovered and available throughout CHAP
- **Two metric types**: Deterministic (point forecasts) and Probabilistic (all samples)

## Quick Start

Here's a minimal deterministic metric:

```python
from chap_core.assessment.metrics.base import (
    AggregationOp,
    DeterministicMetric,
    MetricSpec,
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
from chap_core.assessment.metrics.base import AggregationOp, MetricSpec

spec = MetricSpec(
    metric_id="unique_id",              # Used in APIs and registry
    metric_name="Display Name",          # Human-readable name
    aggregation_op=AggregationOp.MEAN,   # MEAN, SUM, or ROOT_MEAN_SQUARE
    description="What this metric measures",
)
```

## Complete Examples

### Example: RMSE-style Metric

```python
from chap_core.assessment.metrics.base import (
    AggregationOp,
    DeterministicMetric,
    MetricSpec,
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
    )
    low_pct = 10
    high_pct = 90
```

## Using Metrics

### Getting Metrics from the Registry

```python
from chap_core.assessment.metrics import get_metric, get_metrics_registry, list_metrics

# Get a specific metric by ID
MAEClass = get_metric("mae")
mae_metric = MAEClass()
print(f"Metric name: {mae_metric.get_name()}")

# List all available metrics
for info in list_metrics():
    print(f"  {info['id']}: {info['name']}")
```

### Computing Metrics at Different Aggregation Levels

```python
from chap_core.assessment.metrics import get_metric
from chap_core.assessment.flat_representations import DataDimension

# Get the MAE metric
mae = get_metric("mae")()

# These methods are available on all metrics:
# - get_global_metric(obs, forecasts) -> single aggregated value
# - get_detailed_metric(obs, forecasts) -> per location/time/horizon
# - get_metric(obs, forecasts, dimensions=(...)) -> custom aggregation
```

## Registration and Discovery

### The @metric() Decorator

The decorator registers your metric class when the module is imported:

```python
from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import DeterministicMetric, MetricSpec, AggregationOp


@metric()  # This registers the class in the global registry
class RegisteredMetric(DeterministicMetric):
    spec = MetricSpec(
        metric_id="registered_example",
        metric_name="Registered Example",
        aggregation_op=AggregationOp.MEAN,
        description="Example of a registered metric",
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
)
from chap_core.assessment.flat_representations import DataDimension
```

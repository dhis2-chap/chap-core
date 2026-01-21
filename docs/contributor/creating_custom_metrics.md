# Creating Custom Metrics

This guide explains how to create custom evaluation metrics for CHAP backtest results using the metrics system.

## Overview

Metrics in CHAP measure how well a model's forecasts match observed values. The metrics system provides:

- **Single definition**: Each metric is defined once and supports multiple aggregation levels
- **Multi-level aggregation**: Get global values, per-location, per-horizon, or detailed breakdowns
- **Automatic registration**: Metrics are discovered and available throughout CHAP
- **Two metric types**: Deterministic (point forecasts) and Probabilistic (all samples)

Each metric:
- Receives flat pandas DataFrames containing observations and forecasts
- Returns a DataFrame with dimension columns plus a `metric` column
- Is automatically registered and discoverable via the `@metric()` decorator

## Data Formats

Your metric will receive data in standardized DataFrame formats:

### Observations DataFrame (FlatObserved)

Contains the actual observed values:

| Column | Type | Description |
|--------|------|-------------|
| `location` | str | Location identifier |
| `time_period` | str | Time period (e.g., "2024-01" or "2024W01") |
| `disease_cases` | float | Observed disease cases |

### Forecasts DataFrame (FlatForecasts)

Contains forecast samples:

| Column | Type | Description |
|--------|------|-------------|
| `location` | str | Location identifier |
| `time_period` | str | Time period being forecasted |
| `horizon_distance` | int | How many periods ahead this forecast is |
| `sample` | int | Sample index (for probabilistic forecasts) |
| `forecast` | float | Forecasted value |

### Output Format

Metrics return a DataFrame with:
- Zero or more dimension columns (`location`, `time_period`, `horizon_distance`)
- A `metric` column containing the computed values

## Creating a Basic Metric

### Step 1: Choose Your Base Class

CHAP provides two base classes:

- **`DeterministicMetric`**: For metrics that use the median of forecast samples (point forecast comparison)
- **`ProbabilisticMetric`**: For metrics that need all forecast samples

### Step 2: Define the Spec

Create a `MetricSpec` with:

```python
from chap_core.assessment.metrics.base import (
    AggregationOp,
    MetricSpec,
)

spec = MetricSpec(
    metric_id="my_metric",           # Unique identifier (used in APIs)
    metric_name="My Metric",          # Human-readable display name
    aggregation_op=AggregationOp.MEAN,  # How to aggregate: MEAN, SUM, or ROOT_MEAN_SQUARE
    description="Description of what this metric measures",
)
```

### Step 3: Implement the Computation Method

For **deterministic metrics**, implement `compute_point_metric()`:

```console
def compute_point_metric(self, forecast: float, observed: float) -> float:
    """Compute metric for a single forecast/observation pair."""
    return abs(forecast - observed)
```

For **probabilistic metrics**, implement `compute_sample_metric()`:

```console
def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
    """Compute metric from all samples and observation."""
    return float(np.mean(np.abs(samples - observed)))
```

### Step 4: Add the Decorator

Use `@metric()` to register your metric:

```console
from chap_core.assessment.metrics import metric

@metric()
class MyMetric(DeterministicMetric):
    # ...
```

## Complete Examples

### Deterministic Metric: Mean Absolute Error

This example shows a simple deterministic metric that computes absolute error:

```python
from chap_core.assessment.metrics.base import (
    AggregationOp,
    DeterministicMetric,
    MetricSpec,
)
from chap_core.assessment.metrics import metric


@metric()
class MAEMetric(DeterministicMetric):
    """
    Mean Absolute Error metric.

    Computes absolute error at the detailed level. When aggregated using
    MEAN, this produces the MAE (mean of absolute errors).
    """

    spec = MetricSpec(
        metric_id="mae",
        metric_name="MAE",
        aggregation_op=AggregationOp.MEAN,
        description="Mean Absolute Error - measures average absolute prediction error",
    )

    def compute_point_metric(self, forecast: float, observed: float) -> float:
        """Compute absolute error for a single forecast/observation pair."""
        return abs(forecast - observed)
```

### Probabilistic Metric: CRPS

This example shows a probabilistic metric that uses all forecast samples:

```python
import numpy as np

from chap_core.assessment.metrics.base import (
    AggregationOp,
    ProbabilisticMetric,
    MetricSpec,
)
from chap_core.assessment.metrics import metric


@metric()
class CRPSMetric(ProbabilisticMetric):
    """
    Continuous Ranked Probability Score (CRPS) metric.

    CRPS measures both calibration and sharpness of probabilistic forecasts.
    """

    spec = MetricSpec(
        metric_id="crps",
        metric_name="CRPS",
        aggregation_op=AggregationOp.MEAN,
        description="Continuous Ranked Probability Score - measures calibration and sharpness",
    )

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Compute CRPS from all samples and the observation."""
        # CRPS = E[|X - obs|] - 0.5 * E[|X - X'|]
        term1 = np.mean(np.abs(samples - observed))
        term2 = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
        return float(term1 - term2)
```

### Parameterized Metric: Percentile Coverage

For metrics with parameters, create a base class and concrete subclasses:

```python
import numpy as np

from chap_core.assessment.metrics.base import (
    AggregationOp,
    ProbabilisticMetric,
    MetricSpec,
)
from chap_core.assessment.metrics import metric


class PercentileCoverageMetric(ProbabilisticMetric):
    """
    Base class for percentile coverage metrics.

    Computes whether the observation falls within the specified percentile range
    of the forecast samples. Not registered directly - use concrete subclasses.
    """

    low_percentile: int
    high_percentile: int

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Check if observation is within the percentile range."""
        low, high = np.percentile(samples, [self.low_percentile, self.high_percentile])
        return 1.0 if (low <= observed <= high) else 0.0


@metric()
class Coverage10_90Metric(PercentileCoverageMetric):
    """10th-90th percentile coverage metric."""

    spec = MetricSpec(
        metric_id="coverage_10_90",
        metric_name="Coverage 10-90",
        aggregation_op=AggregationOp.MEAN,
        description="Proportion of observations within 10th-90th percentile",
    )
    low_percentile = 10
    high_percentile = 90


@metric()
class Coverage25_75Metric(PercentileCoverageMetric):
    """25th-75th percentile coverage metric."""

    spec = MetricSpec(
        metric_id="coverage_25_75",
        metric_name="Coverage 25-75",
        aggregation_op=AggregationOp.MEAN,
        description="Proportion of observations within 25th-75th percentile",
    )
    low_percentile = 25
    high_percentile = 75
```

## Registration and Discovery

### How Registration Works

The `@metric()` decorator registers your metric class in a global registry when the module is imported. The decorator:

1. Validates that your class inherits from `Metric`
2. Reads the `metric_id` from the class's `spec` attribute
3. Adds the class to the registry under that ID

### Making CHAP Discover Your Metric

For CHAP to discover your metric at startup, import your module in the `_discover_metrics()` function in `chap_core/assessment/metrics/__init__.py`:

```console
def _discover_metrics():
    """Import all metric modules to trigger registration."""
    from chap_core.assessment.metrics import rmse  # noqa: F401
    from chap_core.assessment.metrics import mae  # noqa: F401
    from chap_core.assessment.metrics import crps  # noqa: F401
    # ... existing imports ...
    from chap_core.assessment.metrics import my_custom_metric  # Add your module here
```

### Where to Place Your Metric File

Create your metric file in `chap_core/assessment/metrics/`. For example:
- `chap_core/assessment/metrics/my_custom_metric.py`

## Understanding Aggregation

### Aggregation Operations

The `AggregationOp` enum defines how detailed values are combined:

| Operation | Description |
|-----------|-------------|
| `MEAN` | Average of values (most common) |
| `SUM` | Sum of values |
| `ROOT_MEAN_SQUARE` | Square root of mean of squared values |

### Data Dimensions

The `DataDimension` enum defines the available dimensions:

| Dimension | Description |
|-----------|-------------|
| `location` | Geographic location |
| `time_period` | Time period of the forecast |
| `horizon_distance` | How far ahead the forecast is |

### Using get_metric()

The `get_metric()` method allows you to specify which dimensions to keep:

```console
from chap_core.assessment.flat_representations import DataDimension

metric = MAEMetric()

# Global aggregate (single value)
global_df = metric.get_global_metric(observations, forecasts)
# or equivalently:
global_df = metric.get_metric(observations, forecasts, dimensions=())

# Detailed (per location/time/horizon)
detailed_df = metric.get_detailed_metric(observations, forecasts)

# Per location only
per_loc_df = metric.get_metric(
    observations, forecasts,
    dimensions=(DataDimension.location,)
)

# Per location and horizon
per_loc_horizon_df = metric.get_metric(
    observations, forecasts,
    dimensions=(DataDimension.location, DataDimension.horizon_distance)
)
```

## Testing Your Metric

### Unit Testing

Write a test for your metric in `tests/evaluation/`:

```console
def test_my_custom_metric(flat_observations, flat_forecasts):
    """Test my custom metric with flat data."""
    from chap_core.assessment.metrics.my_custom_metric import MyCustomMetric

    metric = MyCustomMetric()

    # Test global metric
    global_result = metric.get_global_metric(flat_observations, flat_forecasts)
    assert len(global_result) == 1
    assert "metric" in global_result.columns
    assert global_result["metric"].iloc[0] >= 0  # Adjust based on your metric

    # Test detailed metric
    detailed_result = metric.get_detailed_metric(flat_observations, flat_forecasts)
    assert "location" in detailed_result.columns
    assert "time_period" in detailed_result.columns
    assert "horizon_distance" in detailed_result.columns
    assert "metric" in detailed_result.columns
```

The `flat_observations` and `flat_forecasts` fixtures are defined in `tests/evaluation/conftest.py`.

### Manual Testing

You can test your metric interactively:

```console
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics.my_custom_metric import MyCustomMetric

# Load an evaluation from file
evaluation = Evaluation.from_file("example_data/example_evaluation.nc")
flat_data = evaluation.to_flat()

# Test your metric
metric = MyCustomMetric()
result = metric.get_global_metric(flat_data.observations, flat_data.forecasts)
print(f"Global {metric.get_name()}: {result['metric'].iloc[0]:.4f}")
```

## Using Your Metric

### In Code

Once registered, your metric is available via the registry:

```console
from chap_core.assessment.metrics import (
    get_metric,
    get_metrics_registry,
    list_metrics,
    available_metrics,
)

# Get a specific metric class
MetricClass = get_metric("my_metric_id")
metric = MetricClass()

# Get all registered metrics
registry = get_metrics_registry()

# List metrics with metadata
for info in list_metrics():
    print(f"{info['id']}: {info['name']} - {info['description']}")

# Backward compatibility: available_metrics dict
metric_class = available_metrics["my_metric_id"]
```

### In Backtest Evaluation

Metrics are computed on-demand when requested through the REST API or CLI. When visualizing backtest results, the system retrieves the registered metrics and computes them from the stored forecasts and observations.

## Reference

### Existing Implementations

Study these existing implementations as examples:

| File | Description |
|------|-------------|
| `mae.py` | Simple deterministic metric (absolute error) |
| `rmse.py` | Deterministic metric with ROOT_MEAN_SQUARE aggregation |
| `crps.py` | Probabilistic metric using all samples |
| `percentile_coverage.py` | Parameterized probabilistic metric with subclasses |

### API Reference

#### `@metric()` Decorator

```console
@metric()
class MyMetric(Metric):
    spec = MetricSpec(...)
    # ...
```

Registers a metric class in the global registry using `spec.metric_id`.

#### `MetricSpec` Dataclass

```console
MetricSpec(
    metric_id: str,                    # Required: Unique identifier
    metric_name: str,                  # Required: Display name
    aggregation_op: AggregationOp = AggregationOp.MEAN,  # How to aggregate
    description: str = "No description provided",        # Description
)
```

#### `DeterministicMetric` Base Class

For metrics that operate on the median of samples:

```console
class MyMetric(DeterministicMetric):
    spec = MetricSpec(...)

    def compute_point_metric(self, forecast: float, observed: float) -> float:
        """Compute metric for a single forecast/observation pair."""
        pass
```

#### `ProbabilisticMetric` Base Class

For metrics that need all samples:

```console
class MyMetric(ProbabilisticMetric):
    spec = MetricSpec(...)

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Compute metric from all samples and observation."""
        pass
```

#### Helper Functions

```console
# Get all registered metrics
from chap_core.assessment.metrics import get_metrics_registry
registry = get_metrics_registry()

# Get a specific metric class
from chap_core.assessment.metrics import get_metric
metric_cls = get_metric("mae")

# List all metrics with metadata
from chap_core.assessment.metrics import list_metrics
metrics = list_metrics()
```

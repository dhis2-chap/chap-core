# Creating Custom Threshold Strategies

This guide explains how to add a new threshold ("endemic channel") calculation strategy
using the threshold plugin system.

## Overview

A threshold strategy turns a dataset's historical `disease_cases` observations into one
threshold value per requested `(period_id, location)`. Chap provides a registry — mirroring
the backtest plot and metric registries — so new strategies can be added without touching
the endpoint code.

Each strategy:

- Receives a flat pandas DataFrame of historical observations and the periods to score
- Returns one threshold per `(period_id, org_unit)`
- Is automatically registered and discoverable
- Is exposed through the REST API at `POST /v1/analytics/thresholds` and listed by
  `GET /v1/analytics/thresholds/strategies` once registered

## Data Schemas

### Historical observations DataFrame (input)

| Column | Type | Description |
|--------|------|-------------|
| `location` | str | Org unit identifier |
| `time_period` | str | Time period (e.g. `"2024-01"`) |
| `disease_cases` | float | Observed disease cases |

### Result DataFrame (output)

| Column | Type | Description |
|--------|------|-------------|
| `period_id` | str | Period the threshold applies to |
| `org_unit` | str | Org unit the threshold applies to |
| `threshold` | float | Computed threshold value |

## Writing a strategy

Subclass `ThresholdStrategyBase`, implement `compute()`, and register the class with the
`@threshold(...)` decorator:

```python
import pandas as pd

from chap_core.assessment.thresholds import threshold
from chap_core.assessment.thresholds.base import ThresholdStrategyBase


@threshold(
    "historical_percentile",
    "Historical percentile",
    "Threshold as the given percentile of historical same-month values.",
)
class HistoricalPercentileStrategy(ThresholdStrategyBase):
    def compute(
        self,
        historical_observations: pd.DataFrame,
        period_ids: list[str],
        params: dict | None = None,
    ) -> pd.DataFrame:
        q = float((params or {}).get("percentile", 0.95))
        ...  # return DataFrame with columns [period_id, org_unit, threshold]
```

`params` carries optional, strategy-specific parameters supplied in the request body. See
`chap_core/assessment/thresholds/seasonal.py` for the built-in seasonal mean + k*std strategy.

## Registering for discovery

The `@threshold` decorator registers your class in a global registry when its module is
imported. For Chap to discover the strategy at startup, import your module in
`_discover_strategies()` in `chap_core/assessment/thresholds/__init__.py`:

```python
def _discover_strategies():
    from chap_core.assessment.thresholds import (  # noqa: F401
        historical_percentile,
        seasonal,
    )
```

Once registered, the strategy id can be passed as `strategy` to
`POST /v1/analytics/thresholds` and appears in `GET /v1/analytics/thresholds/strategies`.

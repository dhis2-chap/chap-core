# Creating Custom Threshold Strategies

This guide explains how to add a new threshold ("endemic channel") calculation strategy
using the threshold plugin system.

## Overview

A threshold strategy turns a dataset's historical `disease_cases` observations into one or
more threshold lines per requested `(period_id, location)`. Chap provides a registry —
mirroring the backtest plot and metric registries — so new strategies can be added without
touching the endpoint code.

Each strategy:

- Declares a typed pydantic params model whose `type` literal selects the strategy in requests
- Receives a flat pandas DataFrame of historical observations, the periods to score, and its
  validated params model
- Returns one threshold per `(period_id, org_unit, line)`, where `line` indexes the requested
  threshold lines (e.g. one per requested quantile)
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
| `location` | str | Org unit the threshold applies to |
| `line` | int | Zero-based index into the requested line parameter list |
| `threshold` | float | Computed threshold value |

## Writing a strategy

Declare a params model with a `type` literal matching the strategy id, subclass
`ThresholdStrategyBase`, implement `compute()`, and register the class with the
`@threshold(...)` decorator:

```python
from typing import Literal

import pandas as pd
from pydantic import Field

from chap_core.assessment.thresholds import threshold
from chap_core.assessment.thresholds.base import ThresholdStrategyBase
from chap_core.database.base_tables import DBModel


class HistoricalPercentileParams(DBModel):
    type: Literal["historical_percentile"] = "historical_percentile"
    percentile: float = Field(0.95, ge=0.0, le=1.0, description="Percentile of historical same-month values.")


@threshold(
    "historical_percentile",
    "Historical percentile",
    HistoricalPercentileParams,
    "Threshold as the given percentile of historical same-month values.",
)
class HistoricalPercentileStrategy(ThresholdStrategyBase[HistoricalPercentileParams]):
    def compute(
        self,
        historical_observations: pd.DataFrame,
        period_ids: list[str],
        params: HistoricalPercentileParams,
    ) -> pd.DataFrame:
        q = params.percentile
        ...  # return DataFrame with columns [period_id, location, line, threshold]
```

The decorator asserts that the params model's `type` literal matches the registered strategy
id, and binds the model as `params_model` on the class. Pydantic field constraints
(`ge`, `le`, ...) validate requests, so invalid parameters become 422 responses with
field-level messages, and every field's `description` shows up in the OpenAPI schema. See
`chap_core/assessment/thresholds/seasonal.py` and
`chap_core/assessment/thresholds/percentile.py` for the built-in strategies, including how
to support a list-valued line parameter that produces one threshold line per entry.

## Registering for discovery

The `@threshold` decorator registers your class in a global registry when its module is
imported. For Chap to discover the strategy at startup, import your module in
`_discover_strategies()` in `chap_core/assessment/thresholds/__init__.py`:

```python
def _discover_strategies():
    from chap_core.assessment.thresholds import (  # noqa: F401
        historical_percentile,
        percentile,
        seasonal,
    )
```

Finally, add the params model to the `ThresholdParams` discriminated union in
`chap_core/assessment/thresholds/params.py`. The union is what
`POST /v1/analytics/thresholds` accepts as `params` — the `type` field selects your
strategy — and the strategy appears in `GET /v1/analytics/thresholds/strategies`.

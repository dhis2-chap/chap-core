"""Typed per-strategy parameter models for the thresholds API.

Each strategy declares a pydantic model whose ``type`` literal doubles as the
discriminator selecting the strategy, so the OpenAPI schema renders
:data:`ThresholdParams` as a tagged union that generated clients can build a
strategy picker and per-strategy params form from. The :func:`~chap_core.assessment.thresholds.threshold`
registry decorator asserts each strategy's model literal matches its registered id.

Line parameters (``std_multiplier``, ``quantile``) accept a scalar or a list;
each list entry produces one threshold line in the response, computed over the
same historical window.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from chap_core.database.base_tables import DBModel

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class SeasonalParams(DBModel):
    """Parameters for the seasonal mean + k*std strategy."""

    type: Literal["seasonal"] = "seasonal"
    std_multiplier: float | Annotated[list[float], Field(min_length=1)] = Field(
        2.0,
        description="Number of standard deviations above the seasonal mean. "
        "A list produces one threshold line per entry.",
    )


class PercentileParams(DBModel):
    """Parameters for the seasonal percentile (WHO endemic channel) strategy."""

    type: Literal["percentile"] = "percentile"
    quantile: Fraction | Annotated[list[Fraction], Field(min_length=1)] = Field(
        0.75,
        description="Percentile of historical same-season values, as a fraction in [0, 1]. "
        "A list produces one threshold line per entry, e.g. `[0.25, 0.75]` for the endemic channel band.",
    )
    baseline_years: int | None = Field(
        5,
        ge=1,
        description="Number of complete years before the requested periods to compute the baseline from. "
        "`null` uses all available history.",
    )


ThresholdParams = Annotated[SeasonalParams | PercentileParams, Field(discriminator="type")]


def line_values(scalar_or_list: float | list[float]) -> list[float]:
    """Normalize a line parameter to the list of per-line values."""
    if isinstance(scalar_or_list, list):
        return scalar_or_list
    return [scalar_or_list]

"""Typed per-strategy parameter models for the thresholds API.

Each strategy declares a pydantic model whose required ``type`` literal is the
discriminator selecting the strategy, so the OpenAPI schema renders
:data:`ThresholdParams` as a tagged union that generated clients can build a
strategy picker and per-strategy params form from. ``type`` has no default on
purpose: a default would make it optional in the schema, and an optional
discriminator cannot tell the union members apart in generated code. The
:func:`~chap_core.assessment.thresholds.threshold` registry decorator asserts
each strategy's model literal matches its registered id.

Line parameters (``std_multiplier``, ``quantile``) accept a scalar or a list;
each list entry produces one threshold line in the response, computed over the
same historical window. Every model exposes the normalized list as
:attr:`ThresholdParamsBase.lines`, which the strategies compute from and the
endpoint echoes so clients can label lines without knowing the strategy.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from chap_core.database.base_tables import DBModel

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class ThresholdParamsBase(DBModel):
    """Common interface of the per-strategy params models."""

    @property
    def lines(self) -> list[float]:
        """The line parameter as a list, one value per threshold line, in request order."""
        raise NotImplementedError


class SeasonalParams(ThresholdParamsBase):
    """Parameters for the seasonal mean + k*std strategy."""

    type: Literal["seasonal"]
    std_multiplier: float | Annotated[list[float], Field(min_length=1)] = Field(
        2.0,
        description="Number of standard deviations above the seasonal mean. "
        "A list produces one threshold line per entry.",
    )

    @property
    def lines(self) -> list[float]:
        return line_values(self.std_multiplier)


class PercentileParams(ThresholdParamsBase):
    """Parameters for the seasonal percentile (WHO endemic channel) strategy."""

    type: Literal["percentile"]
    quantile: Fraction | Annotated[list[Fraction], Field(min_length=1)] = Field(
        0.75,
        description="Percentile of historical same-season values, as a fraction in [0, 1]. "
        "A list produces one threshold line per entry, e.g. `[0.25, 0.75]` for the endemic channel band.",
    )
    baseline_years: int | None = Field(
        5,
        ge=1,
        description="Number of the most recent complete years in the dataset to compute the baseline from. "
        "A partial final year is excluded. `null` uses all available history.",
    )

    @property
    def lines(self) -> list[float]:
        return line_values(self.quantile)


ThresholdParams = Annotated[SeasonalParams | PercentileParams, Field(discriminator="type")]


def line_values(scalar_or_list: float | list[float]) -> list[float]:
    """Normalize a line parameter to the list of per-line values."""
    if isinstance(scalar_or_list, list):
        return scalar_or_list
    return [scalar_or_list]

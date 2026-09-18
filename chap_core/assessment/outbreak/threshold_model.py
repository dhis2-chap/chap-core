"""Predictive outbreak model: the fraction of forecast samples above the channel."""

from __future__ import annotations

import pandas as pd

from chap_core.assessment.outbreak import outbreak_model
from chap_core.assessment.outbreak.base import PROBABILITY_COLUMNS, OutbreakModelBase


@outbreak_model(
    "threshold",
    "Forecast exceedance",
    "Alert probability as the fraction of a model's forecast samples above the epidemic channel.",
)
class ThresholdOutbreakModel(OutbreakModelBase):
    """Reads a predictive model's samples and counts how many clear the channel.

    The only genuinely probabilistic outbreak model: with a well-spread posterior
    the returned probability is a real probability of the labelled event, which is
    what makes Brier and log score proper against it.
    """

    def alert_probabilities(
        self,
        historical_observations: pd.DataFrame,
        target_periods: list[str],
        *,
        origin_period: str | None = None,
        forecasts: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if forecasts is None:
            raise ValueError("ThresholdOutbreakModel scores forecast samples; `forecasts` is required.")

        empty = pd.DataFrame(columns=PROBABILITY_COLUMNS)
        if forecasts.empty or not target_periods:
            return empty

        thresholds = self.thresholds(historical_observations, target_periods, origin_period=origin_period)
        if thresholds.empty:
            return empty

        scored = forecasts.merge(thresholds, on=["location", "time_period"], how="inner").dropna(subset=["threshold"])
        if scored.empty:
            return empty

        scored["exceeds"] = (scored["forecast"] > scored["threshold"]).astype(float)
        grouped = scored.groupby(["location", "time_period", "horizon_distance"], as_index=False).agg(
            threshold=("threshold", "first"),
            probability=("exceeds", "mean"),
        )
        return pd.DataFrame(grouped[PROBABILITY_COLUMNS])

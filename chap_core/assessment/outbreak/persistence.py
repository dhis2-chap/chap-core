"""Persistence-of-anomaly outbreak model: is the origin period itself running hot?

Standing in June and asked about November, this model ignores forecasts entirely.
It compares this June against the channel for Junes, and alerts on November if
June cleared it -- on the reasoning that a year running above its seasonal norm
tends to keep running above it.

That makes it the null model a forecast pipeline has to beat: skill against
climatology only asks whether a model beat the base rate, while skill against
this asks whether it beat looking out of the window.

It emits a probability of 0 or 1, never anything between, so it is scored on the
classification metrics and Brier -- not log score, where a confident miss is
unbounded and the clipping constant would decide the number rather than the model.
"""

from __future__ import annotations

import pandas as pd

from chap_core.assessment.flat_representations import horizon_diff
from chap_core.assessment.outbreak import outbreak_model
from chap_core.assessment.outbreak.base import PROBABILITY_COLUMNS, OutbreakModelBase


@outbreak_model(
    "persistence_of_anomaly",
    "Persistence of anomaly",
    "Alerts on a target period when the origin period's own observation cleared the channel for its season.",
)
class PersistenceOutbreakModel(OutbreakModelBase):
    """Carries the origin period's anomaly forward to every target period."""

    def alert_probabilities(
        self,
        historical_observations: pd.DataFrame,
        target_periods: list[str],
        *,
        origin_period: str | None = None,
        forecasts: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if origin_period is None:
            raise ValueError(
                "PersistenceOutbreakModel reads the observation at the origin; `origin_period` is required."
            )

        empty = pd.DataFrame(columns=PROBABILITY_COLUMNS)
        if historical_observations.empty or not target_periods:
            return empty

        # The channel the origin observation is judged against, and the channels
        # for the targets -- both from a baseline that ends at the origin.
        origin_thresholds = self.thresholds(historical_observations, [str(origin_period)], origin_period=origin_period)
        target_thresholds = self.thresholds(historical_observations, target_periods, origin_period=origin_period)
        if origin_thresholds.empty or target_thresholds.empty:
            return empty

        at_origin = historical_observations[historical_observations["time_period"].astype(str) == str(origin_period)][
            ["location", "disease_cases"]
        ]
        if at_origin.empty:
            return empty

        signal = at_origin.merge(origin_thresholds[["location", "threshold"]], on="location", how="inner").dropna(
            subset=["threshold"]
        )
        if signal.empty:
            return empty
        signal["probability"] = (signal["disease_cases"] > signal["threshold"]).astype(float)

        scored = target_thresholds.merge(signal[["location", "probability"]], on="location", how="inner").dropna(
            subset=["threshold"]
        )
        if scored.empty:
            return empty

        scored["horizon_distance"] = [horizon_diff(str(period), str(origin_period)) for period in scored["time_period"]]
        return pd.DataFrame(scored[PROBABILITY_COLUMNS])

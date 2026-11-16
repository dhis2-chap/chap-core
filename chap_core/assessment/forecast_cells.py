"""Forecasts as one row per cell, the form metrics are computed on.

A cell is one (location, time_period, horizon_distance) forecast. Every cell has the
same number of samples, so the samples of all cells form one (cells x samples) matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CELL_KEYS = ["location", "time_period", "horizon_distance"]


class MissingSamplesError(ValueError):
    """Raised when forecast cells do not all have the same number of samples."""


@dataclass
class ForecastCells:
    """
    The forecast cells of an evaluation with their samples and observations.

    Attributes:
        keys: One row per cell with ``location``, ``time_period`` and ``horizon_distance``,
            sorted by those columns
        samples: The samples of each cell, shape (cells, samples)
        observed: The observed value of each cell, NaN where there is none
    """

    keys: pd.DataFrame
    samples: np.ndarray
    observed: np.ndarray

    def __len__(self) -> int:
        return len(self.keys)

    @classmethod
    def from_arrays(
        cls,
        location: np.ndarray,
        time_period: np.ndarray,
        horizon_distance: np.ndarray,
        samples: np.ndarray,
        observations: pd.DataFrame,
    ) -> ForecastCells:
        """Cells from per-cell key arrays and a (cells x samples) matrix, observed from flat observations."""
        keys = pd.DataFrame(
            {
                "location": np.asarray(location, dtype=object).astype(str),
                "time_period": np.asarray(time_period, dtype=object).astype(str),
                "horizon_distance": np.asarray(horizon_distance, dtype=np.int64),
            }
        )
        samples = np.asarray(samples, dtype=float)
        if not len(keys):
            return cls(keys=keys, samples=samples.reshape(0, 0), observed=np.empty(0))
        order = keys.sort_values(CELL_KEYS, kind="stable").index.to_numpy()
        keys = keys.iloc[order].reset_index(drop=True)
        return cls(keys=keys, samples=samples[order], observed=_observed_for(keys, observations))

    @classmethod
    def from_flat(cls, observations: pd.DataFrame, forecasts: pd.DataFrame) -> ForecastCells:
        """
        Cells from flat forecasts (one row per sample) and flat observations.

        Rows with a missing key or a missing forecast value are left out.

        Raises:
            MissingSamplesError: if the cells do not all have the same number of samples
        """
        forecasts = forecasts.dropna(subset=[*CELL_KEYS, "forecast"])
        if forecasts.empty:
            return cls.from_arrays(np.empty(0), np.empty(0), np.empty(0), np.empty((0, 0)), observations)
        forecasts = forecasts.sort_values(CELL_KEYS, kind="stable", ignore_index=True)
        group_ids = forecasts.groupby(CELL_KEYS, sort=False).ngroup().to_numpy()
        starts = np.flatnonzero(np.diff(group_ids, prepend=-1))
        sizes = np.diff(np.append(starts, len(forecasts)))
        check_sample_counts(sizes)
        first = forecasts.loc[starts, CELL_KEYS]
        return cls.from_arrays(
            first["location"].to_numpy(),
            first["time_period"].to_numpy(),
            first["horizon_distance"].to_numpy(),
            forecasts["forecast"].to_numpy(dtype=float).reshape(len(starts), int(sizes[0])),
            observations,
        )

    def observed_only(self) -> ForecastCells:
        """The cells that have an observation, which are the ones a metric scores."""
        keep = ~np.isnan(self.observed)
        return ForecastCells(
            keys=self.keys[keep].reset_index(drop=True), samples=self.samples[keep], observed=self.observed[keep]
        )


def check_sample_counts(sizes: np.ndarray) -> None:
    """Raise unless every forecast cell has the same number of samples."""
    if len(sizes) and not (sizes == sizes[0]).all():
        counts = np.unique(sizes)
        raise MissingSamplesError(
            f"Forecasts must all have the same number of samples, but they have between {counts.min()} and "
            f"{counts.max()} ({int((sizes != sizes.max()).sum())} of {len(sizes)} have fewer than the most). "
            "This usually means the model returned NaN for some samples."
        )


def _observed_for(keys: pd.DataFrame, observations: pd.DataFrame) -> np.ndarray:
    if keys.empty or observations.empty:
        return np.full(len(keys), np.nan)
    observed = (
        observations[["location", "time_period", "disease_cases"]]
        .astype({"location": str, "time_period": str})
        .drop_duplicates(["location", "time_period"])
    )
    merged = keys[["location", "time_period"]].merge(observed, on=["location", "time_period"], how="left")
    return merged["disease_cases"].to_numpy(dtype=float)

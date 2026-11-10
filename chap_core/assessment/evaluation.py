"""
Evaluation abstraction for model evaluation results.

This module provides a database-agnostic interface for working with model
evaluation results, enabling better code reuse between REST API and CLI workflows.
"""

from __future__ import annotations

import datetime
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import xarray as xr
from packaging.version import Version

from chap_core.api_types import BacktestParams
from chap_core.assessment.flat_representations import (
    FlatForecasts,
    FlatObserved,
    convert_backtest_observations_to_flat_observations,
    convert_backtest_to_flat_forecasts,
)
from chap_core.assessment.weather_providers import (
    DEFAULT_WEATHER_PROVIDER_ID,
    LEGACY_WEATHER_PROVIDER_ID,
)
from chap_core.database.base_tables import utc_now
from chap_core.database.dataset_tables import DataSet, Observation, ObservationBase
from chap_core.database.tables import Backtest, BacktestForecast, BacktestSpecification
from chap_core.external.ExtendedPredictor import ExtendedPredictor
from chap_core.hpo.types import FlatHyperparameterOptimization, HyperparameterOptimization
from chap_core.rest_api.data_models import BacktestCreate
from chap_core.time_period import Month, TimePeriod

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandera.pandas as pa

    from chap_core.assessment.prediction_evaluator import Estimator
    from chap_core.data import DataSet as _DataSet
    from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB
    from chap_core.datatypes import SamplesWithTruth
    from chap_core.external.model_configuration import ModelTemplateConfigV2
    from chap_core.hpo.meta_learner import MetaLearner

logger = logging.getLogger(__name__)

try:
    from chap_core import __version__ as CHAP_VERSION
except ImportError:
    CHAP_VERSION = "unknown"


def _recorded_version(version: str | None) -> str | None:
    """The version to persist: a real release version, or None for the "unknown" a dev checkout reports."""
    return None if version in (None, "unknown") else version


# Version stored on every backtest this process produces.
RECORDED_CHAP_VERSION = _recorded_version(CHAP_VERSION)


# Version 2 stores forecasts on (location, split, horizon_distance, sample), with the
# forecast period as a ``target_period`` coordinate on (split, horizon_distance).
# Version 1 files (no ``split`` dimension) stored them on (location, time_period,
# horizon_distance, sample), mostly NaN because each split covers only a few of the
# (time_period, horizon_distance) cells (CLIM-1838).
EVALUATION_FORMAT_VERSION = 2


def _split_period(period: str, horizon_distance: int) -> str:
    """The split (last seen) period of a forecast for ``period`` at a 1-based horizon distance."""
    parsed = TimePeriod.parse(period)
    return str((parsed - (horizon_distance - 1) * parsed.time_delta).id)


def _target_periods(splits: list[str], horizons: list[int]) -> np.ndarray:
    """The forecast period for each (split, horizon_distance) cell, as strings."""
    target = np.empty((len(splits), len(horizons)), dtype=object)
    for i, split in enumerate(splits):
        parsed = TimePeriod.parse(split)
        for j, horizon in enumerate(horizons):
            target[i, j] = (parsed + (horizon - 1) * parsed.time_delta).id
    return target


def _forecast_dataarray(
    values: np.ndarray,
    locations: list[str],
    splits: list[str],
    horizons: list[int],
    samples: list[int],
    target_periods: np.ndarray | None = None,
) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("location", "split", "horizon_distance", "sample"),
        coords={
            "location": np.asarray(locations, dtype=object),
            "split": np.asarray(splits, dtype=object),
            "horizon_distance": np.asarray(horizons, dtype=np.int64),
            "sample": np.asarray(samples, dtype=np.int64),
            "target_period": (
                ("split", "horizon_distance"),
                _target_periods(splits, horizons) if target_periods is None else target_periods,
            ),
        },
        name="forecast",
    )


def _observed_dataarray(observations_df: pd.DataFrame, period_dim: str = "time_period") -> xr.DataArray:
    indexed = observations_df.set_index(["location", "time_period"]).rename_axis(index={"time_period": period_dim})
    return indexed["disease_cases"].to_xarray()


def _flat_forecasts_to_dataarray(forecasts_df: pd.DataFrame) -> xr.DataArray:
    """Index flat forecast rows by split instead of time period, so the array holds no padding."""
    if forecasts_df.empty:
        return _forecast_dataarray(np.empty((0, 0, 0, 0)), [], [], [], [])
    time_period = forecasts_df["time_period"].astype(str).to_numpy()
    horizon = forecasts_df["horizon_distance"].astype(np.int64).to_numpy()
    pairs = pd.MultiIndex.from_arrays([time_period, horizon]).unique()
    pair_split = np.array([_split_period(str(p), int(h)) for p, h in pairs], dtype=object)
    split = pair_split[pairs.get_indexer(pd.MultiIndex.from_arrays([time_period, horizon]))]

    location = forecasts_df["location"].astype(str).to_numpy()
    sample = forecasts_df["sample"].astype(np.int64).to_numpy()
    axes = [pd.Index(np.unique(column)) for column in (location, split, horizon, sample)]
    values = np.full(tuple(len(a) for a in axes), np.nan)
    codes = tuple(
        a.get_indexer(pd.Index(column)) for a, column in zip(axes, (location, split, horizon, sample), strict=True)
    )
    values[codes] = forecasts_df["forecast"].to_numpy(dtype=float)
    # Keep the period strings as given; cells no split forecasts get computed ones.
    target = _target_periods(axes[1].tolist(), axes[2].tolist())
    pair_codes = (axes[1].get_indexer(pd.Index(pair_split)), axes[2].get_indexer(pairs.get_level_values(1)))
    target[pair_codes] = np.asarray(pairs.get_level_values(0), dtype=object)
    locations, splits, horizons, samples = (a.tolist() for a in axes)
    return _forecast_dataarray(values, locations, splits, horizons, samples, target_periods=target)


def _flat_data_to_xarray(flat_data: FlatEvaluationData) -> xr.Dataset:
    """
    Convert FlatEvaluationData to an xarray.Dataset in the split-indexed layout.

    Args:
        flat_data: FlatEvaluationData containing forecasts and observations

    Returns:
        xarray.Dataset with ``forecast`` on (location, split, horizon_distance, sample) and
        ``observed`` on (location, time_period)
    """
    data_vars = {
        "forecast": _flat_forecasts_to_dataarray(pd.DataFrame(cast("pd.DataFrame", flat_data.forecasts))),
        "observed": _observed_dataarray(pd.DataFrame(cast("pd.DataFrame", flat_data.observations))),
    }
    if flat_data.historical_observations is not None:
        historical_df = pd.DataFrame(cast("pd.DataFrame", flat_data.historical_observations))
        if not historical_df.empty:
            # Different dimension name, to avoid conflicts with test period observations
            data_vars["historical_observed"] = _observed_dataarray(historical_df, "historical_time_period")
    return xr.Dataset(data_vars)


def _file_attrs(model_metadata: dict, hpo: FlatHyperparameterOptimization | None) -> dict:
    attrs = {
        "title": "CHAP Model Evaluation Results",
        "evaluation_format_version": EVALUATION_FORMAT_VERSION,
        "model_name": model_metadata.get("model_name", ""),
        "model_configuration": json.dumps(model_metadata.get("model_configuration", {})),
        "model_version": model_metadata.get("model_version", ""),
        "created_date": datetime.datetime.now().isoformat(),
        "split_periods": json.dumps(model_metadata.get("split_periods", [])),
        "org_units": json.dumps(model_metadata.get("org_units", [])),
        "historical_context_periods": model_metadata.get("historical_context_periods", 0),
        "future_weather_provider": model_metadata.get("future_weather_provider", DEFAULT_WEATHER_PROVIDER_ID),
        "chap_version": CHAP_VERSION,
    }
    if "model_info" in model_metadata:
        attrs["model_info"] = model_metadata["model_info"]
    if hpo is not None:
        attrs["hpo"] = json.dumps(asdict(hpo))
    return attrs


def _flat_forecasts_from_split_layout(forecast: xr.DataArray) -> pd.DataFrame:
    """Flat forecast rows from a split-indexed forecast array, skipping empty cells."""
    forecast = forecast.transpose("location", "split", "horizon_distance", "sample")
    values = np.asarray(forecast.values, dtype=float)
    flat = values.reshape(-1)
    present = np.flatnonzero(~np.isnan(flat))
    loc_i, split_i, horizon_i, sample_i = np.unravel_index(present, values.shape)
    locations = np.asarray(forecast["location"].values).astype(str).astype(object)
    target = np.asarray(forecast["target_period"].transpose("split", "horizon_distance").values).astype(str)
    return pd.DataFrame(
        {
            "location": locations[loc_i],
            "time_period": target.astype(object)[split_i, horizon_i],
            "horizon_distance": np.asarray(forecast["horizon_distance"].values, dtype=np.int64)[horizon_i],
            "sample": np.asarray(forecast["sample"].values, dtype=np.int64)[sample_i],
            "forecast": flat[present],
        }
    )


def _flat_forecasts_from_period_layout(forecast: xr.DataArray) -> pd.DataFrame:
    """Flat forecast rows from a version 1 file, indexed by time period."""
    forecasts_df = forecast.to_dataframe().reset_index().dropna(subset=["forecast"])
    # NetCDF may have stored these as int32
    forecasts_df["horizon_distance"] = forecasts_df["horizon_distance"].astype(np.int64)
    forecasts_df["sample"] = forecasts_df["sample"].astype(np.int64)
    return forecasts_df


def _flat_observed(observed: xr.DataArray, period_dim: str = "time_period") -> pd.DataFrame:
    df = observed.to_dataframe(name="disease_cases").reset_index()
    df = df.rename(columns={period_dim: "time_period"})[["location", "time_period", "disease_cases"]]
    df["location"] = df["location"].astype(str)
    df["time_period"] = df["time_period"].astype(str)
    return df.dropna(subset=["disease_cases"])


def _xarray_to_flat_data(ds: xr.Dataset) -> FlatEvaluationData:
    """
    Convert xarray.Dataset back to FlatEvaluationData.

    Reads both the split-indexed layout and the time-period-indexed layout of older files.

    Args:
        ds: xarray.Dataset with forecast and observed variables

    Returns:
        FlatEvaluationData with reconstructed forecasts and observations
    """
    if "split" in ds["forecast"].dims:
        forecasts_df = _flat_forecasts_from_split_layout(ds["forecast"])
    else:
        forecasts_df = _flat_forecasts_from_period_layout(ds["forecast"])

    # Load historical observations if present (backwards compatible)
    historical_observations = None
    if "historical_observed" in ds:
        historical_df = _flat_observed(ds["historical_observed"], "historical_time_period")
        if not historical_df.empty:
            historical_observations = FlatObserved.validate(historical_df)

    # Load HPO data if present (backwards compatible)
    hpo = None
    if "hpo" in ds.attrs:
        hpo = FlatHyperparameterOptimization(**json.loads(ds.attrs["hpo"]))

    return FlatEvaluationData(
        forecasts=FlatForecasts.validate(forecasts_df),
        observations=FlatObserved.validate(_flat_observed(ds["observed"])),
        historical_observations=historical_observations,
        hpo=hpo,
    )


def _max_horizon_distance(forecast: xr.DataArray) -> int | None:
    """Largest horizon distance with any forecast in a split-indexed array, or None if empty."""
    present = forecast.notnull().any(dim=[d for d in forecast.dims if d != "horizon_distance"]).values
    if not present.any():
        return None
    return int(forecast["horizon_distance"].values[present].max())


def _historical_observations_frame(dataset: _DataSet, up_to_period: TimePeriod, n_periods: int) -> pd.DataFrame:
    """Up to ``n_periods`` observed disease cases per location, ending at ``up_to_period``, as flat rows."""
    period_ids_by_range: dict[tuple[str, int], list[str]] = {}
    locations, periods, values = [], [], []
    for location in dataset.keys():  # noqa: SIM118
        location_data = dataset[location]
        time_periods = location_data.time_period
        # Locations usually share one period range, so its ids are only listed once.
        key = (time_periods[0].id, len(time_periods)) if len(time_periods) else ("", 0)
        if key not in period_ids_by_range:
            period_ids_by_range[key] = [period.id for period in time_periods]
        period_ids = period_ids_by_range[key]
        end = sum(1 for period_id in period_ids if period_id <= up_to_period.id)
        if end == 0:
            continue
        start = max(0, end - n_periods)
        cases = np.asarray(location_data.disease_cases[start:end], dtype=float)
        keep = ~np.isnan(cases)
        locations.extend([location] * int(keep.sum()))
        periods.extend(np.asarray(period_ids[start:end], dtype=object)[keep].tolist())
        values.extend(cases[keep].tolist())
    return pd.DataFrame({"location": locations, "time_period": periods, "disease_cases": values})


def _observations_frame(observations: list[Observation] | pd.DataFrame | None) -> pd.DataFrame | None:
    if observations is None:
        return None
    if isinstance(observations, pd.DataFrame):
        return observations
    return convert_backtest_observations_to_flat_observations(cast("list[ObservationBase]", observations))


def _samples_with_truth_to_xarray(evaluation_results: Iterable[_DataSet[SamplesWithTruth]]) -> xr.Dataset:
    """Forecast samples and observed cases of each backtest split, in the split-indexed layout.

    Each split forecasts from the first period of its result. An observation seen in
    several splits keeps the value from the first.
    """
    splits: list[str] = []
    split_periods: list[list[str]] = []
    split_entries: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    for eval_result in evaluation_results:
        period_ids = [period.id for period in eval_result.period_range]
        entries = {}
        for location, samples_with_truth in eval_result.items():
            samples = np.asarray(samples_with_truth.samples, dtype=float)
            cases = np.asarray(samples_with_truth.disease_cases, dtype=float)
            n = min(len(period_ids), len(samples), len(cases))
            entries[str(location)] = (samples[:n].reshape(n, -1), cases[:n])
        splits.append(period_ids[0])
        split_periods.append(period_ids)
        split_entries.append(entries)

    locations = sorted({location for entries in split_entries for location in entries})
    n_horizons = max((len(s) for e in split_entries for s, _ in e.values()), default=0)
    n_samples = max((s.shape[1] for e in split_entries for s, _ in e.values()), default=0)
    periods = sorted({period for ids in split_periods for period in ids})
    loc_index = {location: i for i, location in enumerate(locations)}
    period_index = {period: i for i, period in enumerate(periods)}

    forecast = np.full((len(locations), len(splits), n_horizons, n_samples), np.nan)
    observed = np.full((len(locations), len(periods)), np.nan)
    seen = np.zeros(observed.shape, dtype=bool)
    for split_i, (period_ids, entries) in enumerate(zip(split_periods, split_entries, strict=True)):
        for location, (samples, cases) in entries.items():
            loc_i = loc_index[location]
            forecast[loc_i, split_i, : samples.shape[0], : samples.shape[1]] = samples
            cols = np.array([period_index[p] for p in period_ids[: len(cases)]], dtype=np.int64)
            new = ~seen[loc_i, cols]
            observed[loc_i, cols[new]] = cases[new]
            seen[loc_i, cols[new]] = True

    observed_da = xr.DataArray(
        observed,
        dims=("location", "time_period"),
        coords={"location": np.asarray(locations, dtype=object), "time_period": np.asarray(periods, dtype=object)},
        name="observed",
    )
    forecast_da = _forecast_dataarray(
        forecast, locations, splits, list(range(1, n_horizons + 1)), list(range(n_samples))
    )
    return xr.Dataset({"forecast": forecast_da, "observed": observed_da})


@dataclass
class FlatEvaluationData:
    """
    Container for flat representations of evaluation data.

    Combines forecasts and observations which are always used together
    for metric computation and visualization.

    Attributes:
        forecasts: Flat representation of forecast samples
        observations: Flat representation of observed values (test periods for evaluation)
        historical_observations: Flat representation of historical observed values
            before split periods (for plotting context). Optional for backwards compatibility.
    """

    forecasts: pa.typing.DataFrame[FlatForecasts]
    observations: pa.typing.DataFrame[FlatObserved]
    historical_observations: pa.typing.DataFrame[FlatObserved] | None = None
    hpo: FlatHyperparameterOptimization | None = None


class EvaluationBase(ABC):
    """
    Abstract base class for evaluation results.

    An Evaluation represents the complete results of evaluating a model:
    - Forecasts (with samples/quantiles)
    - Observations (ground truth)
    - Metadata (locations, split periods)

    This abstraction is database-agnostic and can be implemented by
    different concrete classes (database-backed, in-memory, etc.).
    """

    @abstractmethod
    def to_flat(self) -> FlatEvaluationData:
        """
        Export evaluation data as flat representations.

        Returns:
            FlatEvaluationData containing FlatForecasts and FlatObserved objects
        """

    @abstractmethod
    def get_org_units(self) -> list[str]:
        """
        Get list of locations included in this evaluation.

        Returns:
            List of location identifiers (org_units)
        """

    @abstractmethod
    def get_split_periods(self) -> list[str]:
        """
        Get list of train/test split periods used in evaluation.

        Returns:
            List of period identifiers (e.g., ["2024-01", "2024-02"])
        """

    @classmethod
    @abstractmethod
    def from_backtest(cls, backtest: Backtest) -> EvaluationBase:
        """
        Create Evaluation from database Backtest object.

        All implementations must support loading from database.

        Args:
            backtest: Database Backtest object (with relationships loaded)

        Returns:
            Evaluation instance
        """

    @classmethod
    @abstractmethod
    def from_samples_with_truth(
        cls,
        evaluation_results: Iterable[_DataSet[SamplesWithTruth]],
        last_train_period: TimePeriod,
        configured_model: ConfiguredModelDB,
        info: BacktestCreate,
        historical_observations: list[Observation] | None = None,
        historical_context_periods: int = 0,
    ) -> EvaluationBase: ...


class Evaluation(EvaluationBase):
    """
    Evaluation results, held either as arrays or as a database Backtest.

    The CLI paths (``create``, ``from_samples_with_truth`` and ``from_file``) keep the
    forecasts and observations in an ``xr.Dataset`` and only build ORM objects when
    ``to_backtest()`` asks for them. ``from_backtest`` wraps an existing database row.
    """

    def __init__(
        self,
        backtest: Backtest,
        historical_observations: list[Observation] | None = None,
        historical_context_periods: int = 0,
        hpo: HyperparameterOptimization | None = None,
    ):
        """
        Initialize Evaluation with a Backtest object.

        Args:
            backtest: Database Backtest object (with relationships loaded)
            historical_observations: Optional list of Observation objects for historical
                context (periods before split points, for plotting)
            historical_context_periods: Number of periods of historical context stored
        """
        self._backtest = backtest
        self._historical_observations = historical_observations or []
        self._historical_context_periods = historical_context_periods
        self._hpo = hpo
        self._flat_data_cache: FlatEvaluationData | None = None
        # Set when the results are held as arrays. The backtest is then a header without
        # forecasts, filled in from the arrays the first time to_backtest() is called.
        self._dataset: xr.Dataset | None = None
        self._backtest_pending = False
        self._last_train_period: str | None = None
        self._in_memory_dataset_name = "cli_evaluation_dataset"

    @classmethod
    def _from_dataset(
        cls,
        backtest: Backtest,
        dataset: xr.Dataset,
        historical_context_periods: int,
        hpo: HyperparameterOptimization | None = None,
        last_train_period: str | None = None,
        in_memory_dataset_name: str = "cli_evaluation_dataset",
    ) -> Evaluation:
        evaluation = cls(backtest, historical_context_periods=historical_context_periods, hpo=hpo)
        evaluation._dataset = dataset
        evaluation._backtest_pending = True
        evaluation._last_train_period = last_train_period
        evaluation._in_memory_dataset_name = in_memory_dataset_name
        backtest.max_horizon_distance = _max_horizon_distance(dataset["forecast"])
        return evaluation

    @property
    def future_weather_provider(self) -> str:
        """Id of the future-weather provider these results were produced with.

        The backtest's specification is the single source of truth, so database
        persistence and NetCDF exports cannot disagree.
        """
        return self._backtest.future_weather_provider

    @classmethod
    def from_backtest(cls, backtest: Backtest) -> Evaluation:
        """
        Create Evaluation from database Backtest object.

        Args:
            backtest: Database Backtest object (with relationships loaded)

        Returns:
            Evaluation instance wrapping the Backtest
        """
        return cls(backtest)

    @classmethod
    def from_samples_with_truth(
        cls,
        evaluation_results: Iterable[_DataSet[SamplesWithTruth]],
        last_train_period: TimePeriod,
        configured_model: ConfiguredModelDB,
        info: BacktestCreate,
        historical_observations: list[Observation] | pd.DataFrame | None = None,
        historical_context_periods: int = 0,
        specification: BacktestSpecification | None = None,
        hpo: HyperparameterOptimization | None = None,
    ) -> Evaluation:
        info.created = utc_now()
        # The parameters live on the specification now and are computed fields on
        # Backtest, reading through the link. Excluding them here keeps that explicit;
        # passing them would be silently ignored rather than rejected.
        backtest = Backtest(
            **info.model_dump(exclude=set(BacktestParams.model_fields))
            | {
                "model_db_id": configured_model.id,
                "model_template_version": configured_model.model_template.version,
                "chap_version": RECORDED_CHAP_VERSION,
            }
        )
        # Callers persisting the backtest pass the deduplicated row resolved against the
        # database. The CLI paths build an in-memory evaluation only, so a detached
        # specification carrying the same parameters is enough for them.
        backtest.specification = specification or BacktestSpecification(
            dataset_id=info.dataset_id, **info.model_dump(include=set(BacktestParams.model_fields))
        )
        dataset = _samples_with_truth_to_xarray(evaluation_results)
        historical_df = _observations_frame(historical_observations)
        if historical_df is not None and not historical_df.empty:
            dataset["historical_observed"] = _observed_dataarray(historical_df, "historical_time_period")

        backtest.org_units = [str(location) for location in dataset["forecast"]["location"].values]
        backtest.split_periods = [str(split) for split in dataset["forecast"]["split"].values]
        return cls._from_dataset(
            backtest,
            dataset,
            historical_context_periods=historical_context_periods,
            hpo=hpo,
            last_train_period=last_train_period.id,
        )

    @classmethod
    def create(
        cls,
        configured_model: ConfiguredModelDB,
        estimator: Estimator | MetaLearner,
        dataset: _DataSet,
        backtest_params: BacktestParams,
        backtest_name: str = "evaluation",
        historical_context_years: int = 6,
        info: BacktestCreate | None = None,
        specification: BacktestSpecification | None = None,
    ) -> Evaluation:
        """
        Uses ``train_test_generator`` to create an expanding window split of the
        data. Create an Evaluation by running a backtest.

        Factory method that handles the complete backtest workflow:
        1. Run backtest with provided estimator
        2. Create Evaluation from results

        Args:
            configured_model: Configured model database object with metadata
            estimator: Model estimator instance ready for training/prediction. It is run
                as given: wrap a model whose max_prediction_periods is below
                ``backtest_params.n_periods`` with ``ExtendedPredictor.extend_to_horizon``
                first. Only the model tuned from a HyperparameterOptimizer is wrapped here.
            dataset: Dataset to evaluate on
            backtest_params: Backtest execution parameters (n_periods, n_splits, stride)
            backtest_name: Name for the backtest (default: "evaluation")
            historical_context_years: Years of historical data to include for plotting
                context (default: 6). Number of periods is calculated based on dataset
                period type (e.g., 6 years = 312 weeks or 72 months).
            info: Backtest metadata to record. Defaults to an in-memory backtest named
                ``backtest_name`` (``dataset_id=0``), as used by the CLI.
            specification: Specification row to attach, for callers that persist the backtest.

        Returns:
            Evaluation instance with backtest results
        """
        from chap_core.assessment.dataset_splitting import train_test_generator
        from chap_core.assessment.prediction_evaluator import backtest
        from chap_core.hpo.hyperparameter_optimizer import HyperparameterOptimizer
        from chap_core.hpo.meta_learner import MetaLearner

        train_set, test_generator = train_test_generator(
            dataset=dataset,
            prediction_length=backtest_params.n_periods,
            n_test_sets=backtest_params.n_splits,
            stride=backtest_params.stride,
            future_weather_provider=backtest_params.future_weather_provider,
        )

        hpo_data = None
        if isinstance(estimator, HyperparameterOptimizer):
            hpo_data = estimator.meta_learn(train_set)
            model = estimator.model_template.get_model(
                hpo_data.model_configuration,  # type: ignore[arg-type]
                prediction_length=backtest_params.n_periods,
            )
            # The tuned model is built here, so callers cannot extend its horizon themselves.
            tuned_estimator = ExtendedPredictor.extend_to_horizon(model(), backtest_params.n_periods)  # type: ignore[assignment]
        elif isinstance(estimator, MetaLearner):
            raise TypeError(f"Unsupported MetaLearner: {type(estimator).__name__}")
        else:
            tuned_estimator = estimator

        # Run backtest
        evaluation_results = backtest(
            estimator=tuned_estimator,
            train_set=train_set,
            test_generator=test_generator,
            n_test_sets=backtest_params.n_splits,
            n_retrain=backtest_params.n_retrain,
        )

        # Prepare metadata
        last_train_period = train_set.period_range[-1]

        backtest_info = info or BacktestCreate(
            name=backtest_name,
            dataset_id=0,
            model_id=configured_model.id,
            n_periods=backtest_params.n_periods,
            n_splits=backtest_params.n_splits,
            stride=backtest_params.stride,
            n_retrain=backtest_params.n_retrain,
            future_weather_provider=backtest_params.future_weather_provider,
        )

        # Calculate number of periods based on dataset period type
        historical_context_periods = cls.calculate_periods_from_years(
            dataset=dataset,
            years=historical_context_years,
        )

        # Extract historical observations from the dataset for plotting context
        historical_observations = cls.extract_historical_observations(
            dataset=dataset,
            up_to_period=last_train_period,
            n_periods=historical_context_periods,
        )

        # Create Evaluation
        return cls.from_samples_with_truth(
            evaluation_results=evaluation_results,
            last_train_period=last_train_period,
            configured_model=configured_model,
            info=backtest_info,
            historical_observations=historical_observations,
            historical_context_periods=historical_context_periods,
            specification=specification,
            hpo=hpo_data,
        )

    @classmethod
    def calculate_periods_from_years(cls, dataset: _DataSet, years: int) -> int:
        """
        Calculate number of periods from years based on dataset period type.

        Args:
            dataset: Dataset to get period type from
            years: Number of years

        Returns:
            Number of periods (e.g., 6 years = 312 weeks or 72 months)
        """
        # Get time delta from the first location's data
        first_location = next(iter(dataset.keys()))
        time_periods = dataset[first_location].time_period
        time_delta = time_periods.delta

        # Calculate periods per year based on time delta
        rd = time_delta._relative_delta
        if rd.days == 7 or rd.weeks == 1:
            # Weekly data
            periods_per_year = 52
        elif rd.months == 1:
            # Monthly data
            periods_per_year = 12
        elif rd.days == 1:
            # Daily data
            periods_per_year = 365
        elif rd.years == 1:
            # Yearly data
            periods_per_year = 1
        else:
            # Default to weekly if unknown
            periods_per_year = 52

        return years * periods_per_year

    @classmethod
    def extract_historical_observations(
        cls,
        dataset: _DataSet,
        up_to_period: TimePeriod,
        n_periods: int,
    ) -> pd.DataFrame:
        """
        Extract historical observations from a dataset for plotting context.

        Args:
            dataset: Dataset containing disease_cases time series
            up_to_period: Include observations up to and including this period
            n_periods: Maximum number of periods to include

        Returns:
            Flat observations (location, time_period, disease_cases), without missing values
        """
        return _historical_observations_frame(dataset, up_to_period, n_periods)

    def to_backtest(self) -> Backtest:
        """
        Get the database Backtest object.

        For results held as arrays, this builds the backtest's forecasts (and, outside the
        database, an in-memory dataset with the observations) on the first call.

        Returns:
            Backtest database model
        """
        if self._backtest_pending:
            self._fill_backtest_from_dataset()
            self._backtest_pending = False
        return self._backtest

    def _fill_backtest_from_dataset(self) -> None:
        assert self._dataset is not None
        forecast = self._dataset["forecast"].transpose("location", "split", "horizon_distance", "sample")
        values = np.asarray(forecast.values, dtype=float)
        locations = [str(location) for location in forecast["location"].values]
        splits = [str(split) for split in forecast["split"].values]
        target = np.asarray(forecast["target_period"].transpose("split", "horizon_distance").values).astype(str)
        for loc_i, location in enumerate(locations):
            for split_i, split in enumerate(splits):
                for horizon_i in range(values.shape[2]):
                    cell = values[loc_i, split_i, horizon_i]
                    cell = cell[~np.isnan(cell)]
                    if cell.size == 0:
                        continue
                    self._backtest.forecasts.append(
                        BacktestForecast(
                            period=str(target[split_i, horizon_i]),
                            org_unit=location,
                            last_train_period=self._last_train_period or split,
                            last_seen_period=split,
                            values=cell.tolist(),
                        )
                    )

        # Backtests persisted to the database reach their observations through dataset_id.
        if self._backtest.dataset_id != 0:
            return
        observed = _flat_observed(self._dataset["observed"])
        observations = [
            Observation(period=period, org_unit=location, value=float(value), feature_name="disease_cases")
            for location, period, value in observed.itertuples(index=False)
        ]
        periods = sorted(observed["time_period"].unique().tolist())
        period_type = None
        if periods:
            period_type = "month" if isinstance(TimePeriod.parse(periods[0]), Month) else "week"
        # Populate the period metadata so frontends that sort/format periods (which need a
        # non-null period_type) can render the evaluation.
        self._backtest.dataset = DataSet(
            name=self._in_memory_dataset_name,
            observations=observations,
            period_type=period_type,
            org_units=self._backtest.org_units,
            first_period=periods[0] if periods else None,
            last_period=periods[-1] if periods else None,
        )

    def to_flat(self) -> FlatEvaluationData:
        """
        Export evaluation data as flat tables.

        Results are cached for performance. Repeated calls return the cached result.

        Returns:
            FlatEvaluationData containing forecasts, observations, and historical observations
        """
        if self._flat_data_cache is None:
            if self._dataset is not None:
                flat_data = _xarray_to_flat_data(self._dataset)
                if self._hpo is not None:
                    flat_data.hpo = self._hpo.to_flat()
                self._flat_data_cache = flat_data
            else:
                self._flat_data_cache = self._flat_data_from_backtest()
        return self._flat_data_cache

    def _flat_data_from_backtest(self) -> FlatEvaluationData:
        forecasts_df = convert_backtest_to_flat_forecasts(self._backtest.forecasts)
        observations_df = convert_backtest_observations_to_flat_observations(
            cast("list[ObservationBase]", self._backtest.dataset.observations),
            target_column=self._backtest.specification.target_column,
        )

        # Convert historical observations if present
        historical_observations = None
        if self._historical_observations:
            historical_df = convert_backtest_observations_to_flat_observations(
                cast("list[ObservationBase]", self._historical_observations)
            )
            if not historical_df.empty:
                historical_observations = FlatObserved.validate(historical_df)

        return FlatEvaluationData(
            forecasts=FlatForecasts.validate(forecasts_df),
            observations=FlatObserved.validate(observations_df),
            historical_observations=historical_observations,
            hpo=self._hpo.to_flat() if self._hpo is not None else None,
        )

    def get_org_units(self) -> list[str]:
        """
        Get locations from Backtest metadata.

        Returns:
            List of location identifiers
        """
        return self._backtest.org_units

    def get_split_periods(self) -> list[str]:
        """
        Get split periods from Backtest metadata.

        Returns:
            List of period identifiers
        """
        return self._backtest.split_periods

    def to_file(
        self,
        filepath: str | Path,
        model_name: str | None = None,
        model_configuration: dict | None = None,
        model_version: str | None = None,
        model_info: ModelTemplateConfigV2 | None = None,
    ) -> None:
        """
        Export evaluation to NetCDF file using xarray.

        Args:
            filepath: Path to output NetCDF file
            model_name: Name of the model (optional)
            model_configuration: Model configuration dictionary (optional)
            model_version: Model version string (optional)
            model_info: ModelTemplateConfigV2 object containing model metadata (optional)
        """
        model_metadata = {
            "model_name": model_name or "",
            "model_configuration": model_configuration or {},
            "model_version": model_version or "",
            "split_periods": self.get_split_periods(),
            "org_units": self.get_org_units(),
            "historical_context_periods": self._historical_context_periods,
            "future_weather_provider": self.future_weather_provider,
        }

        if model_info is not None:
            model_metadata["model_info"] = model_info.model_dump_json(exclude_none=True)

        if self._dataset is not None:
            ds = self._dataset.copy()
            hpo = self._hpo.to_flat() if self._hpo is not None else None
        else:
            flat_data = self.to_flat()
            ds = _flat_data_to_xarray(flat_data)
            hpo = flat_data.hpo
        ds.attrs = _file_attrs(model_metadata, hpo)
        if hpo is None and "hpo" in self._dataset_attrs():
            ds.attrs["hpo"] = self._dataset_attrs()["hpo"]
        if self._last_train_period is not None:
            ds.attrs["last_train_period"] = self._last_train_period
        ds.to_netcdf(filepath)

    def _dataset_attrs(self) -> dict:
        return dict(self._dataset.attrs) if self._dataset is not None else {}

    @classmethod
    def from_file(cls, filepath: str | Path) -> Evaluation:
        """
        Load evaluation from NetCDF file.

        Reads the arrays without building database objects; ``to_backtest()`` builds a
        detached Backtest from them when asked. Files written before the split-indexed
        layout are converted on load.

        Args:
            filepath: Path to NetCDF file

        Returns:
            Evaluation instance
        """
        with xr.open_dataset(filepath) as opened:
            ds = cls._ensure_backcompatibility(opened)
            flat_data = None
            if "split" not in ds["forecast"].dims:
                flat_data = _xarray_to_flat_data(ds)
                converted = _flat_data_to_xarray(flat_data)
                converted.attrs = dict(ds.attrs)
                ds = converted
            ds = ds.load()

        split_periods = json.loads(ds.attrs.get("split_periods", "[]"))
        org_units = json.loads(ds.attrs.get("org_units", "[]"))
        historical_context_periods = int(ds.attrs.get("historical_context_periods", 0))

        backtest = Backtest(
            name=f"Loaded from {Path(filepath).name}",
            org_units=org_units,
            split_periods=split_periods,
            forecasts=[],
            dataset_id=0,
            # The version of the writer, not of the process importing the file.
            chap_version=_recorded_version(ds.attrs.get("chap_version")),
            # The stored file carries no parameters beyond the provider, so this is a
            # detached placeholder that keeps the parameter properties readable.
            specification=BacktestSpecification(
                dataset_id=0,
                org_units=org_units,
                future_weather_provider=str(ds.attrs["future_weather_provider"]),
            ),
        )
        last_train_period = ds.attrs.get("last_train_period")
        evaluation = cls._from_dataset(
            backtest,
            ds,
            historical_context_periods=historical_context_periods,
            last_train_period=str(last_train_period) if last_train_period is not None else None,
            in_memory_dataset_name=f"dataset_from_{Path(filepath).name}",
        )
        evaluation._flat_data_cache = flat_data
        if "hpo" in ds.attrs:
            logger.warning(
                "HPO doesn't yet support converting flat representation back into HyperparameterOptimization, "
                "but FlatHyperparameterOptimization will be available in flat data."
            )

        return evaluation

    @staticmethod
    def _ensure_backcompatibility(ds: xr.Dataset) -> xr.Dataset:
        """
        Ensure backwards compatibility for datasets created with older CHAP versions.

        Update horizon_distance coordinate in older datasets where it was stored as 0-based instead of 1-based.

        Fill in future_weather_provider for files written before the provider was
        recorded. Those evaluations were run with the observed weather of each
        forecast window, so they are labelled accordingly rather than inheriting
        today's default.
        """

        if Version(ds.attrs.get("chap_version", "0.0.0")) <= Version("1.1.1"):
            ds = ds.assign_coords(horizon_distance=ds.horizon_distance + 1)
        # A file without the attribute is legacy by definition; gating on the
        # writer's version would break on dev checkouts whose CHAP_VERSION is
        # "unknown" and on files written by a newer release than the reader.
        if "future_weather_provider" not in ds.attrs:
            ds.attrs["future_weather_provider"] = LEGACY_WEATHER_PROVIDER_ID
        return ds

    def get_hpo(self) -> HyperparameterOptimization | None:
        return self._hpo


@dataclass
class ModelCard:
    """
    ModelCard is a class that includes metadata
    for documenting model evaluations, such as descriptions, authorship,
    and usage notes.
    """

    def __init__(
        self,
        backtest: Backtest,
        description: str | None = None,
    ):
        self.backtest = backtest
        self.description = description

from enum import StrEnum
from typing import Any, Literal

import numpy as np
from geojson_pydantic import (
    Feature as _Feature,
)
from geojson_pydantic import (
    FeatureCollection as _FeatureCollection,
)
from geojson_pydantic import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from pydantic import BaseModel, Field, model_validator

from chap_core.database.base_tables import DBModel


class FeatureModel(_Feature):
    """GeoJSON `Feature` carrying one org-unit polygon plus DHIS2-style metadata.

    Loosens upstream `geojson_pydantic.Feature` so `id` may be omitted and
    `geometry` may be any geometry variant or `None`.
    """

    id: str | None = Field(default=None, description="External identifier of the org unit (DHIS2 id), if known.")  # type: ignore[assignment]
    properties: dict[str, Any] | None = Field(
        default_factory=dict, description="Free-form properties attached to the feature."
    )  # type: ignore[assignment]
    geometry: Point | MultiPoint | LineString | MultiLineString | Polygon | MultiPolygon | None = Field(
        default=None,
        description="GeoJSON geometry. Any of the standard variants, or `None` if unknown.",
    )  # type: ignore[assignment]


class FeatureCollectionModel(_FeatureCollection):
    """GeoJSON `FeatureCollection` of `FeatureModel`s — the polygon set for a dataset."""

    features: list[FeatureModel] = Field(description="One feature per org unit in the polygon set.")  # type: ignore[assignment]


class DataElement(BaseModel):
    """Single data point in a v1-format request: a value for one (period, org-unit) pair."""

    pe: str = Field(description="Period identifier (DHIS2-style, e.g. `202403`).")
    ou: str = Field(description="Org-unit identifier.")
    value: float | None = Field(description="Observed value; `None` is allowed for known-missing observations.")


class DataList(BaseModel):
    """Time-series of v1-format `DataElement`s for one feature on one DHIS2 data element."""

    featureId: str = Field(description="Canonical feature identifier (matching a `FeatureType.name`).")
    dhis2Id: str = Field(description="External DHIS2 data element id the values came from.")
    data: list[DataElement] = Field(..., min_length=1, description="At least one (period, org-unit, value) row.")


class DataElementV2(BaseModel):
    """Single data point in a v2-format request — same as `DataElement` but with verbose field names."""

    period: str = Field(description="Period identifier (e.g. `202403`).")
    orgUnit: str = Field(description="Org-unit identifier.")
    value: float | None = Field(description="Observed value; `None` is allowed for known-missing observations.")


class DataListV2(BaseModel):
    """Time-series of v2-format `DataElementV2`s for one feature on one data element."""

    featureId: str = Field(description="Canonical feature identifier (matching a `FeatureType.name`).")
    dataElement: str = Field(description="External data element id the values came from.")
    data: list[DataElementV2] = Field(..., min_length=1, description="At least one (period, org-unit, value) row.")


class RequestV1(BaseModel):
    """V1 request body: GeoJSON polygons plus a list of feature time-series."""

    orgUnitsGeoJson: FeatureCollectionModel = Field(description="GeoJSON polygon set for the org units in the request.")
    features: list[DataList] = Field(description="One time-series per (feature, data-element) pair.")


class RequestV2(RequestV1):
    """V2 request body: a v1 request plus an explicit estimator id."""

    estimator_id: str = Field(default="chap_ewars_monthly", description="Configured-model name to run.")


class PredictionRequest(RequestV2):
    """Prediction request body: a v2 request plus a forecast horizon and a flag to echo back the input data."""

    n_periods: int = Field(default=3, description="Number of periods to forecast.")
    include_data: bool = Field(
        default=False, description="When True, the response echoes the request inputs alongside the predictions."
    )


class PredictionEntry(BaseModel):
    """One quantile-aware predicted value for a (period, org-unit) pair."""

    orgUnit: str = Field(description="Org-unit identifier the prediction is for.")
    period: str = Field(description="Period the prediction is for.")
    quantile: float = Field(description="Quantile of the predictive distribution this value represents (0.0 to 1.0).")
    value: float = Field(description="Predicted value at the given quantile.")


class EvaluationEntry(PredictionEntry):
    """Backtest evaluation entry: a prediction plus the split it was produced under."""

    splitPeriod: str = Field(description="Period at which the rolling split was advanced for this prediction.")


class BacktestParams(DBModel):
    """Shared backtest scheduling parameters used by request models."""

    n_periods: int = Field(default=3, gt=0, description="Number of periods to forecast at each split.")
    n_splits: int = Field(default=7, gt=0, description="Total number of rolling train/test splits.")
    stride: int = Field(default=1, gt=0, description="Number of periods to advance between successive splits.")
    n_retrain: int = Field(
        default=1,
        gt=0,
        description="Number of times the model is retrained, evenly spaced across the splits. 1 means train once.",
    )

    @model_validator(mode="after")
    def _check_n_retrain(self) -> "BacktestParams":
        if self.n_retrain > self.n_splits:
            raise ValueError("n_retrain cannot exceed n_splits.")
        return self


class RunConfig(BaseModel):
    """Configuration for model execution environment."""

    ignore_environment: bool = Field(default=False, description="When True, skip the automatic environment setup step.")
    debug: bool = Field(default=False, description="When True, enable verbose debug logging.")
    log_file: str | None = Field(default=None, description="Path to write log output to; `None` means stdout only.")
    run_directory_type: Literal["latest", "timestamp", "use_existing"] = Field(
        default="timestamp",
        description="How to name the per-run subdirectory under `runs/`.",
    )
    is_chapkit_model: bool = Field(
        default=False,
        description="Set to True when the model is served via chapkit (REST) rather than an MLproject directory.",
    )
    track: bool = Field(default=False, description="When True, log params/metrics/artifacts to the tracking backend.")


class EvaluationResponse(BaseModel):
    """Backtest evaluation response: actual cases plus per-(period, org-unit, quantile) predictions."""

    actualCases: DataList = Field(description="Observed values for the evaluation window, used as the truth signal.")
    predictions: list[EvaluationEntry] = Field(
        description="Per-(period, org-unit, quantile) predictions from every split."
    )

    def model_dump(self, **kwargs):
        """Override to handle special types during serialization"""
        data = super().model_dump(**kwargs)
        return self._clean_for_json(data)

    def _clean_for_json(self, obj):
        """Recursively clean data for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, "id"):
            # Handle period objects with id attribute
            return obj.id
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            # Fallback to string representation
            return str(obj)


class PeriodObservation(BaseModel):
    """Single observation indexed by its period."""

    time_period: str = Field(description="Period identifier the observation is for.")


class EstimatorMode(StrEnum):
    """How the estimator should be run when training/evaluating a model."""

    NORMAL = "normal"
    HPO = "hpo"
    ENSEMBLE = "ensemble"


class SearcherType(StrEnum):
    GRID = "grid"
    RANDOM = "random"
    TPE = "tpe"


class EstimatorOptions(BaseModel):
    """Options controlling how the estimator runs (mode + metric)."""

    mode: EstimatorMode = Field(
        default=EstimatorMode.NORMAL,
        description="Estimator mode: 'normal' = normal run, 'hpo' = hyperparameter optimization, 'ensemble' = ensemble learning.",
    )
    metric: str | None = Field(
        default=None,
        description="Metric used for HPO or ensemble. Default will be used if none provided. Ignored in normal mode.",
    )
    searcher: SearcherType | None = Field(
        default=None,
        description="Searcher used for HPO. If not provided, a default RandomSearcher will be used. Ignored in normal and ensemble modes.",
    )

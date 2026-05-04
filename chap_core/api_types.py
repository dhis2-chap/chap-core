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
from pydantic import BaseModel, Field

from chap_core.database.base_tables import DBModel


class FeatureModel(_Feature):
    id: str | None = None  # type: ignore[assignment]
    properties: dict[str, Any] | None = Field(default_factory=dict)  # type: ignore[assignment]
    geometry: Point | MultiPoint | LineString | MultiLineString | Polygon | MultiPolygon | None = None  # type: ignore[assignment]


class FeatureCollectionModel(_FeatureCollection):
    features: list[FeatureModel]  # type: ignore[assignment]


class DataElement(BaseModel):
    pe: str
    ou: str
    value: float | None


class DataList(BaseModel):
    featureId: str
    dhis2Id: str
    data: list[DataElement] = Field(..., min_length=1)


class DataElementV2(BaseModel):
    period: str
    orgUnit: str
    value: float | None


class DataListV2(BaseModel):
    featureId: str
    dataElement: str
    data: list[DataElementV2] = Field(..., min_length=1)


class RequestV1(BaseModel):
    orgUnitsGeoJson: FeatureCollectionModel
    features: list[DataList]


class RequestV2(RequestV1):
    estimator_id: str = "chap_ewars_monthly"


class PredictionRequest(RequestV2):
    n_periods: int = 3
    include_data: bool = False


class PredictionEntry(BaseModel):
    orgUnit: str
    period: str
    quantile: float
    value: float


class EvaluationEntry(PredictionEntry):
    splitPeriod: str


class BacktestParams(DBModel):
    n_periods: int = 3
    n_splits: int = 7
    stride: int = 1


class RunConfig(BaseModel):
    """Configuration for model execution environment."""

    ignore_environment: bool = False
    debug: bool = False
    log_file: str | None = None
    run_directory_type: Literal["latest", "timestamp", "use_existing"] = "timestamp"
    is_chapkit_model: bool = False


class EvaluationResponse(BaseModel):
    actualCases: DataList
    predictions: list[EvaluationEntry]

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
    time_period: str


class EstimatorMode(StrEnum):
    NORMAL = "normal"
    HPO = "hpo"
    ENSEMBLE = "ensemble"


class EstimatorOptions(BaseModel):
    mode: EstimatorMode = Field(
        default=EstimatorMode.NORMAL,
        description=(
            "Estimator mode: 'normal' = normal run, 'hpo' = hyperparameter optimization, 'ensemble' = ensemble learning."
        ),
    )
    metric: str | None = Field(
        default="rmse",
        description="Metric used for HPO or ensemble. Ignored in normal mode.",
    )

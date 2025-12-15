from typing import Any, Optional, Literal
import numpy as np

from pydantic import BaseModel, Field
from pydantic_geojson import (
    FeatureCollectionModel as _FeatureCollectionModel,
    FeatureModel as _FeatureModel,
)

from chap_core.database.base_tables import DBModel


class FeatureModel(_FeatureModel):
    id: Optional[str] = None
    properties: Optional[dict[str, Any]] = Field(default_factory=dict)


class FeatureCollectionModel(_FeatureCollectionModel):
    features: list[FeatureModel]


class DataElement(BaseModel):
    pe: str
    ou: str
    value: Optional[float]


class DataList(BaseModel):
    featureId: str
    dhis2Id: str
    data: list[DataElement] = Field(..., min_length=1)


class DataElementV2(BaseModel):
    period: str
    orgUnit: str
    value: Optional[float]


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


class BackTestParams(DBModel):
    n_periods: int = 3
    n_splits: int = 7
    stride: int = 1


class RunConfig(BaseModel):
    """Configuration for model execution environment."""

    ignore_environment: bool = False
    debug: bool = False
    log_file: Optional[str] = None
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


# class Geometry:
#     type: str
#     coordinates: list[list[float]]
#
#
# class GeoJSONObject(BaseModel):
#     id: str
#     geometry: dict
#
#
# class GeoJSON(BaseModel):
#     type: str
#     features: list[dict]

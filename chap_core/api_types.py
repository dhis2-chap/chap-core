from typing import Any, Optional

from pydantic import BaseModel, Field
from pydantic_geojson import (
    FeatureCollectionModel as _FeatureCollectionModel,
    FeatureModel as _FeatureModel,
)


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
    data: list[DataElement] = Field(..., min_items=1)


class RequestV1(BaseModel):
    orgUnitsGeoJson: FeatureCollectionModel
    features: list[DataList]


class RequestV2(RequestV1):
    estimator_id: str = "chap_ewars_monthly"


class PredictionRequest(RequestV2):
    n_periods: int = 3


class PredictionEntry(BaseModel):
    orgUnit: str
    period: str
    quantile: float
    value: float


class EvaluationEntry(PredictionEntry):
    splitPeriod: str


class EvaluationResponse(BaseModel):
    actualCases: DataList
    predictions: list[EvaluationEntry]


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

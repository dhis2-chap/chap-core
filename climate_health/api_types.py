from pydantic import BaseModel
from pydantic_geojson import (
    FeatureCollectionModel as _FeatureCollectionModel,
    FeatureModel as _FeatureModel,
)


class FeatureModel(_FeatureModel):
    id: str


class FeatureCollectionModel(_FeatureCollectionModel):
    features: list[FeatureModel]


class DataElement(BaseModel):
    pe: str
    ou: str
    value: float


class DataList(BaseModel):
    featureId: str
    dhis2Id: str
    data: list[DataElement]


class RequestV1(BaseModel):
    orgUnitsGeoJson: FeatureCollectionModel
    features: list[DataList]

class RequestV2(RequestV1):
    model_id: str = 'chap_ewars'

class PredictionRequest(RequestV2):
    n_periods: int = 3


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

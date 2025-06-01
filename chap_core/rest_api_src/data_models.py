from typing import List

from pydantic import BaseModel

from chap_core.api_types import FeatureCollectionModel
from chap_core.database.base_tables import DBModel
from chap_core.database.dataset_tables import DataSetBase, ObservationBase
from chap_core.database.tables import BackTestBase, BackTestMetric, BackTestForecast, BackTestRead


class PredictionBase(BaseModel):
    orgUnit: str
    dataElement: str
    period: str


class PredictionResponse(PredictionBase):
    value: float


class PredictionSamplResponse(PredictionBase):
    values: list[float]


class FullPredictionResponse(BaseModel):
    diseaseId: str
    dataValues: List[PredictionResponse]


class FullPredictionSampleResponse(BaseModel):
    diseaseId: str
    dataValues: List[PredictionSamplResponse]


class FetchRequest(DBModel):
    feature_name: str
    data_source_name: str


class DatasetMakeRequest(DataSetBase):
    geojson: FeatureCollectionModel
    provided_data: List[ObservationBase]
    data_to_be_fetched: List[FetchRequest]


class JobResponse(BaseModel):
    id: str

class ValidationError(DBModel):
    reason: str
    org_unit: str
    feature_name: str
    time_periods: List[str]

class ImportSummaryResponse(DBModel):
    id: str | None
    imported_count: int
    rejected: list[ValidationError]

class BackTestCreate(BackTestBase): ...


class BackTestFull(BackTestRead):
    metrics: list[BackTestMetric]
    forecasts: list[BackTestForecast]

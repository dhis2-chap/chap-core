from pydantic import BaseModel

from chap_core.api_types import FeatureCollectionModel
from chap_core.database.base_tables import DBModel
from chap_core.database.dataset_tables import DataSetCreateInfo, ObservationBase
from chap_core.database.tables import BackTestBase, BackTestForecast, BackTestMetric, BackTestRead


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
    dataValues: list[PredictionResponse]


class FetchRequest(DBModel):
    feature_name: str
    data_source_name: str


class DatasetMakeRequest(DataSetCreateInfo):
    geojson: FeatureCollectionModel
    provided_data: list[ObservationBase]
    data_to_be_fetched: list[FetchRequest]


class JobResponse(BaseModel):
    id: str


class PredictionParams(DBModel):
    model_id: str
    n_periods: int = 3


class ValidationError(DBModel):
    reason: str
    org_unit: str
    feature_name: str
    time_periods: list[str]


class ImportSummaryResponse(DBModel):
    id: str | None
    imported_count: int
    rejected: list[ValidationError]


class BackTestCreate(BackTestBase):
    # Accept either the configured-model integer primary key or its string
    # name. The underlying DB column (`BackTestBase.model_id`) is a string,
    # but the `POST /v1/crud/backtests/` handler resolves an `int` to the
    # corresponding name before persisting so the column stays consistent.
    # See `chap_core.rest_api.v1.routers.crud.create_backtest` for the
    # resolution path.
    model_id: int | str  # type: ignore[assignment]


class BackTestFull(BackTestRead):
    metrics: list[BackTestMetric]
    forecasts: list[BackTestForecast]

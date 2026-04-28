from pydantic import BaseModel

from chap_core.api_types import BackTestParams, FeatureCollectionModel
from chap_core.database.base_tables import DBModel
from chap_core.database.dataset_tables import DataSetCreateInfo, ObservationBase
from chap_core.database.model_templates_and_config_tables import (
    ModelTemplateInformation,
    ModelTemplateMetaData,
)
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


class BacktestDomain(DBModel):
    org_units: list[str]
    split_periods: list[str]


class ChapDataSource(DBModel):
    name: str
    display_name: str
    supported_features: list[str]
    description: str
    dataset: str


class MakePredictionRequest(DatasetMakeRequest, PredictionParams):
    meta_data: dict = {}


class MakeBacktestRequest(BackTestParams):
    name: str
    model_id: str
    dataset_id: int


class MakeBacktestWithDataRequest(DatasetMakeRequest, BackTestParams):
    name: str
    model_id: str


class DataBaseResponse(DBModel):
    id: int


class DatasetCreate(DataSetCreateInfo):
    observations: list[ObservationBase]
    geojson: FeatureCollectionModel


class ModelTemplateRead(DBModel, ModelTemplateInformation, ModelTemplateMetaData):
    """
    ModelTemplateRead is a read model for the ModelTemplateDB.
    It is used to return the model template in a readable format.
    """

    name: str
    id: int
    user_options: dict | None = None
    required_covariates: list[str] = []
    version: str | None = None
    archived: bool = False
    health_status: str | None = None
    uses_chapkit: bool = False


class ModelConfigurationCreate(DBModel):
    name: str
    model_template_id: int
    user_option_values: dict | None = None
    additional_continuous_covariates: list[str] = []


class PredictionCreate(DBModel):
    dataset_id: int
    estimator_id: str
    n_periods: int


class BackTestUpdate(DBModel):
    name: str | None = None

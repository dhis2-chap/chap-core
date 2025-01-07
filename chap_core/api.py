import logging

from .assessment.forecast import forecast as do_forecast
from typing import Optional, List
from .assessment.dataset_splitting import train_test_split_with_weather
from .datatypes import (
    HealthData,
    ClimateData,
    HealthPopulationData,
)
from .external.external_model import get_model_from_directory_or_github_url
from .file_io.example_data_set import DataSetType, datasets
from .plotting.prediction_plot import plot_forecast_from_summaries
from .predictor import get_model
from .spatio_temporal_data.temporal_dataclass import DataSet
import dataclasses

from .time_period.date_util_wrapper import delta_month, Month

from .transformations.covid_mask import mask_covid_data

logger = logging.getLogger(__name__)


class DummyControl:
    def set_status(self, status):
        pass

    @property
    def current_control(self):
        return None


@dataclasses.dataclass
class AreaPolygons:
    shape_file: str


@dataclasses.dataclass
class PredictionData:
    area_polygons: AreaPolygons = None
    health_data: DataSet[HealthData] = None
    climate_data: DataSet[ClimateData] = None
    population_data: DataSet[HealthPopulationData] = None
    disease_id: Optional[str] = None
    features: List[object] = None


def extract_disease_name(health_data: dict) -> str:
    return health_data["rows"][0][0]




def train_with_validation(model_name, dataset_name, n_months=12):
    dataset = datasets[dataset_name].load()
    # assert not np.any(np.any(np.isnan(data.to_array()[:, 1:])) for data in dataset.values()), "Dataset contains NaN values"
    # assert not any(np.any(np.isnan(data.mean_temperature) | np.isnan(data.rainfall)) for data in dataset.values()), "Dataset contains NaN values"
    dataset = mask_covid_data(dataset)
    model = get_model(model_name)(n_iter=32000)
    # split_point = dataset.end_timestamp - n_months * delta_month
    # train_data, test_data, future_weather = train_test_split_with_weather(dataset, split_point)
    prediction_length = n_months * delta_month
    split_point = dataset.end_timestamp - prediction_length
    split_period = Month(split_point.year, split_point.month)
    train_data, test_set, future_weather = train_test_split_with_weather(dataset, split_period)
    model.set_validation_data(test_set)
    model.train(train_data)
    predictions = model.forecast(future_weather, forecast_delta=n_months * delta_month, n_samples=100)
    # plot predictions
    figs = []
    for location, prediction in predictions.items():
        fig = plot_forecast_from_summaries(
            prediction.data(), dataset.get_location(location).data()
        )  # , lambda x: np.log(x+1))
        figs.append(fig)
    return figs


def forecast(
        model_name: str,
        dataset_name: DataSetType,
        n_months: int,
        model_path: Optional[str] = None,
):
    logging.basicConfig(level=logging.INFO)
    dataset = datasets[dataset_name].load()

    if model_name == "external":
        model = get_model_from_directory_or_github_url(model_path)
    else:
        model = get_model(model_name)
        model = model()

    # model = get_model(model_name)()
    predictions = do_forecast(model, dataset, n_months * delta_month)

    figs = []
    for location, prediction in predictions.items():
        fig = plot_forecast_from_summaries(
            prediction.data(), dataset.get_location(location).data()
        )  # , lambda x: np.log(x+1))
        figs.append(fig)
    return figs

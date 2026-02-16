import dataclasses
import logging

from .assessment.forecast import forecast as do_forecast
from .datatypes import (
    ClimateData,
    HealthData,
    HealthPopulationData,
)
from .file_io.example_data_set import DataSetType, datasets
from .models.utils import get_model_from_directory_or_github_url
from .plotting.prediction_plot import plot_forecast_from_summaries
from .predictor import get_model
from .spatio_temporal_data.temporal_dataclass import DataSet
from .time_period.date_util_wrapper import delta_month

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
    area_polygons: AreaPolygons | None = None
    health_data: DataSet[HealthData] | None = None
    climate_data: DataSet[ClimateData] | None = None
    population_data: DataSet[HealthPopulationData] | None = None
    disease_id: str | None = None
    features: list[object] | None = None


def extract_disease_name(health_data: dict) -> str:
    return str(health_data["rows"][0][0])


def forecast(
    model_name: str,
    dataset_name: DataSetType,
    n_months: int,
    model_path: str | None = None,
):
    logging.basicConfig(level=logging.INFO)
    dataset = datasets[dataset_name].load()

    if model_name == "external":
        model = get_model_from_directory_or_github_url(model_path)
    else:
        model_class = get_model(model_name)  # type: ignore[arg-type]
        if model_class is None:
            raise ValueError(f"Model {model_name} not found")
        model = model_class()

    # model = get_model(model_name)()
    predictions = do_forecast(model, dataset, n_months * delta_month)

    figs = []
    for location, prediction in predictions.items():
        fig = plot_forecast_from_summaries(
            prediction.data(), dataset.get_location(location).data()
        )  # , lambda x: np.log(x+1))
        figs.append(fig)
    return figs

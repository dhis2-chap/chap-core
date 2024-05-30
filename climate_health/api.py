import logging
import json
from .assessment.forecast import forecast as do_forecast
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np

from .assessment.dataset_splitting import train_test_split_with_weather
from .datatypes import HealthData, ClimateData, HealthPopulationData, SimpleClimateData, ClimateHealthData, FullData
from .dhis2_interface.json_parsing import predictions_to_datavalue, parse_disease_data, json_to_pandas, \
    parse_population_data
from .external.external_model import get_model_from_yaml_file
from .file_io.example_data_set import DataSetType, datasets
#from .external.external_model import ExternalCommandLineModel, get_model_from_yaml_file
from .geojson import geojson_to_shape, geojson_to_graph
from .plotting.prediction_plot import plot_forecast_from_summaries
from .predictor import get_model
from .spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
import dataclasses

from .time_period.date_util_wrapper import Week, delta_week, delta_month, Month
import git

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
    health_data: SpatioTemporalDict[HealthData] = None
    climate_data: SpatioTemporalDict[ClimateData] = None
    population_data: SpatioTemporalDict[HealthPopulationData] = None


def read_zip_folder(zip_file_path: str) -> PredictionData:
    # read zipfile, create PredictionData
    ziparchive = zipfile.ZipFile(zip_file_path)
    expected_files = {
        "area_polygons": "orgUnits.json",
        "disease": "disease.json",
        "population": "population.json",
        "temperature": "temperature.json",
        "precipitation": "precipitation.json",
    }
    json_data = json.load(ziparchive.open(expected_files["disease"]))
    name_mapping = {
        "time_period": 2,
        "disease_cases": 3,
        "location": 1
    }
    disease = parse_disease_data(json_data, name_mapping=name_mapping)

    temperature_json = json.load(ziparchive.open(expected_files["temperature"]))
    name_mapping = {
        "time_period": 2,
        "mean_temperature": 3,
        "location": 1
    }
    temperature = json_to_pandas(temperature_json, name_mapping)

    precipitation_json = json.load(ziparchive.open(expected_files["temperature"]))
    name_mapping = {
        "time_period": 2,
        "precipitation": 3,
        "location": 1
    }
    precipitation = json_to_pandas(precipitation_json, name_mapping)

    assert np.all(precipitation.time_period == temperature.time_period)
    assert np.all(precipitation.location == temperature.location)

    temperature["rainfall"] = precipitation["precipitation"]
    temperature["rainfall"] = temperature["rainfall"].astype(float)
    temperature["mean_temperature"] = temperature["mean_temperature"].astype(float)
    climate = SpatioTemporalDict.from_pandas(temperature, dataclass=SimpleClimateData)

    population_json = json.load(ziparchive.open(expected_files["population"]))
    population = parse_population_data(population_json)
    graph_file_name = ''
    if False:
        graph_file_name = Path(zip_file_path).with_suffix(".graph")
        area_polygons_file = ziparchive.open(expected_files["area_polygons"])
        geojson_to_graph(area_polygons_file, graph_file_name)
    # geojson_to_shape(area_polygons_file, shape_file_name)

    # geojson_to_shape(str(zip_file_path) + "!area_polygons", shape_file_name)

    return PredictionData(
        health_data=disease,
        climate_data=climate,
        population_data=population,
        area_polygons=AreaPolygons(graph_file_name)
    )

    out_data = {}


#    ...

def get_model_from_directory_or_github_url(model_path, base_working_dir=Path('runs/')):
    """
    Gets the model and initializes a working directory with the code for the model
    """
    is_github = False
    if isinstance(model_path, str) and model_path.startswith("https://github.com"):
        dir_name = model_path.split("/")[-1].replace(".git", "")
        model_name = dir_name
        is_github = True
    else:
        model_name = Path(model_path).name

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    working_dir = base_working_dir / model_name / timestamp

    if is_github:
        working_dir.mkdir(parents=True)
        git.Repo.clone_from(model_path, working_dir)
    else:
        # copy contents of model_path to working_dir
        shutil.copytree(model_path, working_dir)

    # assert that a config file exists
    assert (working_dir / 'config.yml').exists(), f"config.yml file not found in {working_dir}"
    return get_model_from_yaml_file(working_dir / 'config.yml', working_dir)


def get_model_maybe_yaml(model_name):
    if model_name.endswith(".yaml") or model_name.endswith(".yml"):
        working_dir = Path(model_name).parent
        model = get_model_from_yaml_file(model_name, working_dir)
        return model, model.name
    else:
        return get_model(model_name), model_name


def dhis_zip_flow(zip_file_path: str, out_json: Optional[str]=None, model_name=None, n_months=4, docker_filename: Optional[str] = None) -> List[dict] | None:
    data: PredictionData = read_zip_folder(zip_file_path)
    json_body = train_on_prediction_data(data, model_name, n_months, docker_filename)
    if out_json is not None:
        with open(out_json, "w") as f:
            json.dump(json_body, f)
        return None
    else:
        return json_body


def train_on_prediction_data(data, model_name=None, n_months=4, docker_filename=None, control=DummyControl()):
    control.set_status('Preprocessing')
    model, model_name = get_model_maybe_yaml(model_name, docker_filename)
    model = model()
    start_endpoint = min(data.health_data.start_timestamp,
                         data.climate_data.start_timestamp)
    end_endpoint = max(data.health_data.end_timestamp,
                       data.climate_data.end_timestamp)
    new_dict = {}
    for location in data.health_data.locations():
        health = data.health_data.get_location(location).fill_to_range(start_endpoint, end_endpoint)
        climate = data.climate_data.get_location(location).fill_to_range(start_endpoint, end_endpoint)
        assert location in data.population_data, f"Location {location} not in population data: {data.population_data.keys()}"
        population = data.population_data[location]

        new_dict[location] = FullData.combine(health.data(), climate.data(), population)
    climate_health_data = SpatioTemporalDict(new_dict)
    prediction_start = Month(climate_health_data.end_timestamp) - n_months * delta_month
    train_data, _, future_weather = train_test_split_with_weather(climate_health_data, prediction_start)
    logger.info(f"Training model {model_name} on {len(train_data.items())} locations")
    control.set_status('Training')
    if hasattr(model, 'set_training_control'):
        model.set_training_control(control.current_control)

    model.train(climate_health_data)  # , extra_args=data.area_polygons)
    logger.info(f"Forecasting using {model_name} on {len(train_data.items())} locations")
    control.set_status('Forecasting')
    predictions = model.forecast(future_weather, forecast_delta=n_months * delta_month)
    attrs = ['median', 'quantile_high', 'quantile_low']
    logger.info(f"Converting predictions to json")
    control.set_status('Postprocessing')
    data_values = predictions_to_datavalue(predictions, attribute_mapping=dict(zip(attrs, attrs)))
    json_body = [dataclasses.asdict(element) for element in data_values]
    return json_body


def forecast(model_name: str, dataset_name: DataSetType, n_months: int, model_path: Optional[str] = None):
    logging.basicConfig(level=logging.INFO)
    dataset = datasets[dataset_name].load()

    if model_name == 'external':
        model = get_model_from_directory_or_github_url(model_path)
    else:
        model = get_model(model_name)
        model = model()

    # model = get_model(model_name)()
    predictions = do_forecast(model, dataset, n_months * delta_month)

    figs = []
    for location, prediction in predictions.items():
        fig = plot_forecast_from_summaries(prediction.data(),
                                           dataset.get_location(location).data())  # , lambda x: np.log(x+1))
        figs.append(fig)
    return figs

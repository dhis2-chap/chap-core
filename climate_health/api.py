import json
import zipfile
from pathlib import Path
import numpy as np

from climate_health.datatypes import HealthData, ClimateData, HealthPopulationData, SimpleClimateData, ClimateHealthData
from climate_health.dhis2_interface.json_parsing import predictions_to_json, parse_disease_data, json_to_pandas, \
    parse_population_data
from climate_health.external.external_model import ExternalCommandLineModel, get_model_from_yaml_file
from climate_health.geojson import geojson_to_shape
from climate_health.predictor import get_model
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
import dataclasses


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
        "area_polygons": "organisations.geojson",
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

    climate = SpatioTemporalDict.from_pandas(temperature, dataclass=SimpleClimateData)

    population_json = json.load(ziparchive.open(expected_files["population"]))
    population = parse_population_data(population_json)
    shape_file_name = Path(zip_file_path).parent
    geojson_to_shape(ziparchive.open(expected_files["area_polygons"]), shape_file_name)
    #geojson_to_shape(str(zip_file_path) + "!area_polygons", shape_file_name)

    return PredictionData(
        health_data=disease,
        climate_data=climate,
        population_data=population,
        area_polygons=AreaPolygons(shape_file_name)
    )


    out_data = {}




#    ...


def dhis_zip_flow(zip_file_path: str, out_json: str, model_name):
    data: PredictionData = read_zip_folder(zip_file_path)
    model = get_model_from_yaml_file(f'external_models/{model_name}', model_name)
    climate_health_data= ClimateHealthData.combine(data.health_data, data.climate_data)
    model.train(climate_health_data, extra_data=data.area_polygons)
    predictions = model.predict(data)
    predictions_to_json(predictions, out_json)

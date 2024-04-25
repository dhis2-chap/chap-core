import json
import zipfile

from climate_health.datatypes import HealthData, ClimateData, HealthPopulationData
from climate_health.dhis2_interface.json_parsing import predictions_to_json, parse_disease_data
from climate_health.predictor import get_model
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
import dataclasses


@dataclasses.dataclass
class AreaPolygons:
    ...


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


    return PredictionData(
        health_data=disease,
    )


    out_data = {}




#    ...


def dhis_zip_flow(zip_file_path: str, out_json: str, model_name):
    data: PredictionData = read_zip_folder(zip_file_path)
    model = get_model(model_name)
    model.train(data)
    predictions = model.predict(data)
    predictions_to_json(predictions, out_json)

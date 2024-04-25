from climate_health.datatypes import HealthData, ClimateData, HealthPopulationData
from climate_health.dhis2_interface.json_parsing import predictions_to_json
from climate_health.predictor import get_model
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
import dataclasses


@dataclasses.dataclass
class AreaPolygons:
    ...


@dataclasses.dataclass
class PredictionData:
    area_polygons: AreaPolygons
    health_data: SpatioTemporalDict[HealthData]
    climate_data: SpatioTemporalDict[ClimateData]
    population_data: SpatioTemporalDict[HealthPopulationData]



def read_zip_folder(zip_file_path: str):
    #
    zip_file_reader = ZipFileReader(zip_file_path)
    ...


#def convert_geo_json(geo_json_content) -> OurShapeFormat:
#    ...


def dhis_zip_flow(zip_file_path: str, out_json: str, model_name):
    data: PredictionData = read_zip_folder(zip_file_path)
    model = get_model(model_name)
    model.train(data)
    predictions = model.predict(data)
    predictions_to_json(predictions, out_json)

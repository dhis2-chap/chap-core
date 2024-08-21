from dotenv import find_dotenv, load_dotenv
from climate_health.api import read_zip_folder, train_on_prediction_data
from climate_health.api_types import RequestV1
from climate_health.datatypes import FullData, TimeSeriesArray
from climate_health.dhis2_interface.pydantic_to_spatiotemporal import v1_conversion
from climate_health.google_earth_engine.gee_era5 import Era5LandGoogleEarthEngine
from climate_health.predictor import get_model
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


def initialize_gee_client():
    gee_client = Era5LandGoogleEarthEngine()
    return gee_client

def train_on_zip_file(file, model_name, model_path, control=None):
    
    gee_client = initialize_gee_client()
    
    print('train_on_zip_file')
    print('F', file)
    prediction_data = read_zip_folder(file.file)

    prediction_data.climate_data = gee_client.get_historical_era5(prediction_data.features, prediction_data.health_data.period_range)

    return train_on_prediction_data(prediction_data, model_name=model_name, model_path=model_path, control=control)


def train_on_json_data(json_data: RequestV1, model_name, model_path, control=None):
    translations = {'diseases': 'disease_cases'}

    print(json_data)
    json_data = RequestV1.model_validate_json(json_data)
    print(json_data.orgUnitsGeoJson)
    data = {translations.get(feature.featureId, feature.featureId): v1_conversion(feature.data, fill_missing=feature.featureId=='diseases') for feature in json_data.features}
    gee_client = initialize_gee_client()
    period_range = data['disease_cases'].period_range
    climate_data = gee_client.get_historical_era5(json_data.orgUnitsGeoJson.model_dump(), periodes=period_range)
    field_dict = {field_name:
                      SpatioTemporalDict({location: TimeSeriesArray(period_range,  getattr(data, field_name)) for location, data in climate_data.items()})
                  for field_name in ('mean_temperature', 'rainfall')}
    train_data = SpatioTemporalDict.from_fields(FullData, data|field_dict)
    model = get_model(model_name)()

    if hasattr(model, 'set_graph'):
        model.set_graph(data.area_polygons)

    #model.train(train_data)  # , extra_args=data.area_polygons)

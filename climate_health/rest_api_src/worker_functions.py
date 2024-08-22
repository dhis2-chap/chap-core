import json
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from climate_health.api import read_zip_folder, train_on_prediction_data
from climate_health.api_types import RequestV1
from climate_health.climate_data.seasonal_forecasts import SeasonalForecast
from climate_health.datatypes import FullData, TimeSeriesArray, remove_field
from climate_health.dhis2_interface.json_parsing import predictions_to_datavalue
from climate_health.dhis2_interface.pydantic_to_spatiotemporal import v1_conversion
from climate_health.google_earth_engine.gee_era5 import Era5LandGoogleEarthEngine
from climate_health.predictor import get_model
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period import delta_month, PeriodRange, dataclasses


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
    data_path = Path('/home/knut/Data/ch_data/seasonal_forecasts')
    #data_path = Path(__file__).parent.parent.parent / 'example_data' / 'seasonal_forecasts'
    if not data_path.exists():
        raise FileNotFoundError(f'Could not find seasonal forecast data at {data_path}')


    translations = {'diseases': 'disease_cases'}

    #print(json_data)
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
    climate_forecasts = load_forecasts(data_path)

    delta = period_range.delta
    prediction_range = PeriodRange(period_range.end_timestamp,
                                   period_range.end_timestamp + 3 * delta, delta)
    print(period_range)
    print(prediction_range)
    locations = list(climate_data.keys())
    future_weather = {field_name:
                          SpatioTemporalDict({location: climate_forecasts.get_forecasts(location, prediction_range, field_name) for location in locations})
                        for field_name in ('mean_temperature', 'rainfall')}
    future_weather = SpatioTemporalDict.from_fields(remove_field(FullData, 'disease_cases'), future_weather)


    model.train(train_data)  # , extra_args=data.area_polygons)

    predictions = model.forecast(future_weather, forecast_delta=3*delta)
    attrs = ['median', 'quantile_high', 'quantile_low']
    data_values = predictions_to_datavalue(predictions, attribute_mapping=dict(zip(attrs, attrs)))
    json_body = [dataclasses.asdict(element) for element in data_values]
    diseaseId = data.disease_id
    return {'diseaseId': diseaseId, 'dataValues': json_body}


def load_forecasts(data_path):
    climate_forecasts = SeasonalForecast()
    for file_name in os.listdir(data_path):
        print(file_name)
        variable_type = file_name.split('.')[0]
        if file_name.endswith('.json'):
            with open(data_path / file_name) as f:
                climate_forecasts.add_json(variable_type, json.load(f))
    return climate_forecasts




import json
import os
from typing import Optional, Tuple, List

import numpy as np

from chap_core.api import read_zip_folder, train_on_prediction_data
from chap_core.api_types import RequestV1, PredictionRequest, EvaluationEntry, EvaluationResponse
from chap_core.assessment.forecast import forecast_with_predicted_weather, forecast_ahead
from chap_core.assessment.prediction_evaluator import backtest
from chap_core.climate_data.seasonal_forecasts import SeasonalForecast
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData, TimeSeriesArray
from chap_core.dhis2_interface.json_parsing import predictions_to_datavalue
from chap_core.dhis2_interface.pydantic_to_spatiotemporal import v1_conversion
from chap_core.external.external_model import (
    get_model_from_directory_or_github_url,
)
from chap_core.google_earth_engine.gee_era5 import Era5LandGoogleEarthEngine
from chap_core.predictor.model_registry import registry
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import dataclasses
import logging

logger = logging.getLogger(__name__)

# model_paths = {
#    'chap_ewars': 'https://github.com/sandvelab/chap_auto_ewars'
# }


def initialize_gee_client(usecwd=False):
    gee_client = Era5LandGoogleEarthEngine(usecwd=usecwd)
    return gee_client


def train_on_zip_file(file, model_name, model_path, control=None):
    gee_client = initialize_gee_client()

    print("train_on_zip_file")
    print("F", file)
    prediction_data = read_zip_folder(file.file)

    prediction_data.climate_data = gee_client.get_historical_era5(
        prediction_data.features, prediction_data.health_data.period_range
    )

    return train_on_prediction_data(prediction_data, model_name=model_name, model_path=model_path, control=control)


def predict(json_data: PredictionRequest):
    estimator, json_data, target_id, train_data = _convert_prediction_request(json_data)
    predictions = forecast_ahead(estimator, train_data, json_data.n_periods)
    summaries = DataSet({location: samples.summaries() for location, samples in predictions.items()})
    attrs = ["median", "quantile_high", "quantile_low"]
    data_values = predictions_to_datavalue(summaries, attribute_mapping=dict(zip(attrs, attrs)))
    json_body = [dataclasses.asdict(element) for element in data_values]
    return {"diseaseId": target_id, "dataValues": json_body}


def _convert_prediction_request(json_data):
    json_data = PredictionRequest.model_validate_json(json_data)
    estimator = registry.get_model(json_data.estimator_id)
    target_id = get_target_id(json_data, ["disease", "diseases", "disease_cases"])
    train_data = dataset_from_request_v1(json_data)
    return estimator, json_data, target_id, train_data


def evaluate(json_data: PredictionRequest, n_splits: Optional[int]=None, stride: int=1, quantiles: Tuple[float]=(0.25, 0.75))->EvaluationResponse:
    estimator, json_data, target_id, train_data = _convert_prediction_request(json_data)
    print([(data_list.featureId, data_list.dhis2Id) for data_list in json_data.features])
    print(target_id)
    real_data = next(data_list for data_list in json_data.features if data_list.dhis2Id == target_id)
    predictions_list = backtest(estimator, train_data, prediction_length=json_data.n_periods,
                           n_test_sets=n_splits, stride=stride, weather_provider=QuickForecastFetcher)
    evaluation_entries: List[EvaluationEntry]=[]
    for predictions in predictions_list:
        first_period = predictions.period_range[0]
        for location, samples in predictions.items():
            quantiles = {q: np.quantile(samples.samples, q, axis=-1) for q in quantiles}
            for q, quantile in quantiles.items():
                for period, value in zip(predictions.period_range, quantile):
                    entry = EvaluationEntry(orgUnit=location,
                                            period=period.id,
                                            quantile=q,
                                            value=value,
                                            splitPeriod=first_period.id)
                    evaluation_entries.append(entry)

    return EvaluationResponse(actualCases=real_data, predictions=evaluation_entries) # .model_dump()


def train_on_json_data(json_data: RequestV1, model_name, model_path, control=None):
    model_path = model_name
    json_data = RequestV1.model_validate_json(json_data)
    target_name = "diseases"
    target_id = get_target_id(json_data, target_name)
    train_data = dataset_from_request_v1(json_data)
    model = get_model_from_directory_or_github_url(model_path)
    if hasattr(model, "set_graph"):
        logger.warning(f"Not setting graph on {model}")

    predictor = model.train(train_data)  # , extra_args=data.area_polygons)
    predictions = forecast_with_predicted_weather(predictor, train_data, 3)
    summaries = DataSet({location: samples.summaries() for location, samples in predictions.items()})
    attrs = ["median", "quantile_high", "quantile_low"]
    data_values = predictions_to_datavalue(summaries, attribute_mapping=dict(zip(attrs, attrs)))
    json_body = [dataclasses.asdict(element) for element in data_values]

    return {"diseaseId": target_id, "dataValues": json_body}


def get_target_id(json_data, target_names):
    if isinstance(target_names, str):
        target_names = [target_names]
    target_id = next(data_list.dhis2Id for data_list in json_data.features if data_list.featureId in target_names)
    return target_id


def dataset_from_request_v1(
    json_data: RequestV1, target_name="diseases", usecwd_for_credentials=False
) -> DataSet[FullData]:
    translations = {target_name: "disease_cases"}
    data = {
        translations.get(feature.featureId, feature.featureId): v1_conversion(
            feature.data, fill_missing=feature.featureId in (target_name, "population")
        )
        for feature in json_data.features
    }
    gee_client = initialize_gee_client(usecwd=usecwd_for_credentials)
    period_range = data["disease_cases"].period_range
    locations = list(data["disease_cases"].keys())
    climate_data = gee_client.get_historical_era5(json_data.orgUnitsGeoJson.model_dump(), periodes=period_range)
    field_dict = {
        field_name: DataSet(
            {
                location: TimeSeriesArray(period_range, getattr(climate_data[location], field_name))
                for location in locations
            }
        )
        for field_name in ("mean_temperature", "rainfall")
    }
    train_data = DataSet.from_fields(FullData, data | field_dict)
    train_data = train_data.interpolate(["population"])
    return train_data


def load_forecasts(data_path):
    climate_forecasts = SeasonalForecast()
    for file_name in os.listdir(data_path):
        print(file_name)
        variable_type = file_name.split(".")[0]
        if file_name.endswith(".json"):
            with open(data_path / file_name) as f:
                climate_forecasts.add_json(variable_type, json.load(f))
    return climate_forecasts

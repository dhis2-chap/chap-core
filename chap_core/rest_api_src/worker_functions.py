import json
import os
from typing import Optional, Tuple, List, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from chap_core.api_types import RequestV1, PredictionRequest, EvaluationEntry, EvaluationResponse, DataList, \
    DataElement, DataElementV2
from chap_core.assessment.forecast import forecast_with_predicted_weather, forecast_ahead
from chap_core.assessment.prediction_evaluator import backtest
from chap_core.climate_data.seasonal_forecasts import SeasonalForecast
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData, Samples, HealthData, HealthPopulationData, create_tsdataclass, TimeSeriesArray
from chap_core.time_period.date_util_wrapper import convert_time_period_string
from chap_core.external.external_model import (
    get_model_from_directory_or_github_url,
)
from chap_core.google_earth_engine.gee_era5 import Era5LandGoogleEarthEngine
from chap_core.predictor.model_registry import registry
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import dataclasses
import logging

logger = logging.getLogger(__name__)
DISEASE_NAMES = ["disease", "diseases", "disease_cases"]


@dataclasses.dataclass
class DataValue:
    value: int
    orgUnit: str
    dataElement: str
    period: str


class WorkerConfig(BaseModel):
    is_test: bool = False

    # Make it frozen so that we can't accidentally change it
    class Config:
        frozen = True


def initialize_gee_client(usecwd=False, worker_config: WorkerConfig = WorkerConfig()):
    if worker_config.is_test:
        from chap_core.testing.mocks import GEEMock
        return GEEMock()
    gee_client = Era5LandGoogleEarthEngine(usecwd=usecwd)
    return gee_client


def predict_pipeline_from_health_data(health_dataset: DataSet[HealthPopulationData],
                                      estimator_id: str, n_periods: int,
                                      target_id='disease_cases', worker_config: WorkerConfig = WorkerConfig()):
    health_dataset = DataSet.from_dict(health_dataset, HealthPopulationData)
    dataset = harmonize_health_dataset(health_dataset, usecwd_for_credentials=False, worker_config=worker_config)
    estimator = registry.get_model(estimator_id,
                                   ignore_env=estimator_id.startswith('chap_ewars'))
    predictions = forecast_ahead(estimator, dataset, n_periods)
    return sample_dataset_to_prediction_response(predictions, target_id)


def predict_pipeline_from_full_data(dataset: dict,
                                    estimator_id: str, n_periods: int,
                                    target_id='disease_cases', worker_config: WorkerConfig = WorkerConfig()):
    dataset = DataSet.from_dict(dataset, FullData)
    estimator = registry.get_model(estimator_id,
                                   ignore_env=estimator_id.startswith('chap_ewars'))
    predictions = forecast_ahead(estimator, dataset, n_periods)
    return sample_dataset_to_prediction_response(predictions, target_id)


def predict(json_data: PredictionRequest):
    estimator, json_data, target_id, train_data = _convert_prediction_request(json_data)
    predictions = forecast_ahead(estimator, train_data, json_data.n_periods)
    respones = sample_dataset_to_prediction_response(predictions, target_id)
    return respones


def sample_dataset_to_prediction_response(predictions: DataSet[Samples], target_id: str) -> dict:
    summaries = DataSet({location: samples.summaries() for location, samples in predictions.items()})
    attrs = ["median", "quantile_high", "quantile_low"]
    data_values = predictions_to_datavalue(summaries, attribute_mapping=dict(zip(attrs, attrs)))
    json_body = [dataclasses.asdict(element) for element in data_values]
    response = {"diseaseId": target_id, "dataValues": json_body}
    return response


def _convert_prediction_request(json_data: PredictionRequest, worker_config: WorkerConfig = WorkerConfig()):
    json_data = PredictionRequest.model_validate_json(json_data)
    skip_env = hasattr(json_data, "ignore_env") and json_data.ignore_env
    if json_data.estimator_id.startswith('chap_ewars'):
        skip_env = True
        logger.warning(
            f"Hack: Skipping env for {json_data.model_id if hasattr(json_data, 'model_id') else json_data.estimator_id}")

    estimator = registry.get_model(json_data.estimator_id, ignore_env=skip_env)
    target_id = get_target_id(json_data, ["disease", "diseases", "disease_cases"])
    train_data = dataset_from_request_v1(json_data, worker_config=worker_config)
    return estimator, json_data, target_id, train_data


def dataset_to_datalist(dataset: DataSet[HealthData], target_id: str) -> DataList:
    element_list = [DataElement(pe=row.time_period.id, value=row.disease_cases, ou=location) for location, data in
                    dataset.items()
                    for row in data]
    return DataList(dhis2Id=target_id, featureId='disease_cases', data=element_list)


def evaluate(json_data: PredictionRequest, n_splits: Optional[int] = None, stride: int = 1,
             quantiles: Tuple[float] = (0.25, 0.5, 0.75, 0.1, 0.9),
             worker_config: WorkerConfig = WorkerConfig()
             ) -> EvaluationResponse:
    estimator, json_data, target_id, train_data = _convert_prediction_request(json_data, worker_config=worker_config)
    real_data = next(data_list for data_list in json_data.features if data_list.dhis2Id == target_id)
    predictions_list = backtest(estimator, train_data, prediction_length=json_data.n_periods,
                                n_test_sets=n_splits, stride=stride, weather_provider=QuickForecastFetcher)
    return samples_to_evaluation_response(predictions_list, quantiles, real_data)


def __clean_actual_cases(real_data: DataList) -> DataList:
    ''' Temporary function to clean time period names and fill nan valuse to a datalist of real cases'''
    df = pd.DataFrame([{'time_period': row.pe, 'location': row.ou, 'value': row.value} for row in real_data.data])
    print(df['time_period'])
    dataset = DataSet.from_pandas(df, TimeSeriesArray, fill_missing=True)
    return DataList(featureId=real_data.featureId,
                    dhis2Id=real_data.dhis2Id,
                    data=[DataElement(pe=row.time_period.id, ou=location, value=row.value if not np.isnan(row.value) else None)
                          for location, ts_array in dataset.items() for row in ts_array])



def samples_to_evaluation_response(predictions_list, quantiles, real_data: DataList):
    evaluation_entries: List[EvaluationEntry] = []
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
    real_data = __clean_actual_cases(real_data)
    return EvaluationResponse(actualCases=real_data,
                              predictions=evaluation_entries)  # .model_dump()


def train_on_json_data(json_data: RequestV1, model_name, model_path, control=None):
    model_path = model_name
    json_data = PredictionRequest.model_validate_json(json_data)
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


def get_target_name(json_data):
    data_elements = {d.featureId for d in json_data.features}
    possible_target_names = set(DISEASE_NAMES)
    target_name = possible_target_names.intersection(data_elements)
    if not target_name:
        raise ValueError(f"No target name found in {data_elements}")
    if len(target_name) > 1:
        raise ValueError(f"Multiple target names found in {data_elements}: {target_name}")
    return target_name.pop()


def dataset_from_request_v1(
        json_data: RequestV1, target_name="diseases", usecwd_for_credentials=False,
        worker_config: WorkerConfig = WorkerConfig()
) -> DataSet[FullData]:
    dataset = get_health_dataset(json_data)
    return harmonize_health_dataset(dataset, usecwd_for_credentials, worker_config=worker_config)


def harmonize_health_dataset(dataset, usecwd_for_credentials, worker_config: WorkerConfig = WorkerConfig()):
    gee_client = initialize_gee_client(usecwd=usecwd_for_credentials, worker_config=worker_config)
    period_range = dataset.period_range
    climate_data = gee_client.get_historical_era5(dataset.polygons.model_dump(), periodes=period_range)
    train_data = dataset.merge(climate_data, FullData)
    return train_data


def get_health_dataset(json_data: PredictionRequest, dataclass=None, colnames=('ou', 'pe')):
    if dataclass is None:
        dataclass = FullData if hasattr(json_data, 'include_data') and json_data.include_data else HealthPopulationData

    target_name = get_target_name(json_data)
    translations = {target_name: "disease_cases"}
    data = {
        translations.get(feature.featureId, feature.featureId): v1_conversion(
            feature.data, fill_missing=feature.featureId in (target_name, "population"), colnames=colnames
        )
        for feature in json_data.features
    }
    dataset = DataSet.from_fields(dataclass, data)
    dataset = dataset.interpolate(["population"])
    dataset.set_polygons(json_data.orgUnitsGeoJson)
    return dataset


def get_combined_dataset(json_data: RequestV1):
    '''Get a dataset of potentially multiple data types from a RequestV1 object.'''
    dataclass = create_tsdataclass([d.featureId for d in json_data.features])
    return get_health_dataset(json_data, dataclass)


def load_forecasts(data_path):
    climate_forecasts = SeasonalForecast()
    for file_name in os.listdir(data_path):
        print(file_name)
        variable_type = file_name.split(".")[0]
        if file_name.endswith(".json"):
            with open(data_path / file_name) as f:
                climate_forecasts.add_json(variable_type, json.load(f))
    return climate_forecasts


def predictions_to_datavalue(data: DataSet[HealthData], attribute_mapping: dict[str, str]):
    entries = []
    for location, data in data.items():
        data = data.data()
        for i, time_period in enumerate(data.time_period):
            for from_name, to_name in attribute_mapping.items():
                entry = DataValue(
                    getattr(data, from_name)[i],
                    location,
                    to_name,
                    time_period.to_string().replace("-", ""),
                )

                entries.append(entry)
    return entries


def v1_conversion(data_list: list[Union[DataElement, DataElementV2]], fill_missing=False, colnames=('ou', 'pe')) -> \
        DataSet[TimeSeriesArray]:
    """
    Convert a list of DataElement objects to a SpatioTemporalDict[TimeSeriesArray] object.
    """
    location_col, period_col = colnames
    df = pd.DataFrame([d.model_dump() for d in data_list])
    df.sort_values(by=[location_col, period_col], inplace=True)
    d = dict(
        time_period=[convert_time_period_string(row) for row in df[period_col]],
        location=df[location_col],
        value=df.value,
    )
    converted_df = pd.DataFrame(d)
    ds = DataSet.from_pandas(converted_df, TimeSeriesArray, fill_missing=fill_missing)
    return ds

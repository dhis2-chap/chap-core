import dataclasses
import json
import logging
import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from chap_core.api_types import (
    DataElement,
    DataElementV2,
    DataList,
    EvaluationEntry,
    EvaluationResponse,
    PredictionRequest,
    RequestV1,
)
from chap_core.assessment.forecast import forecast_with_predicted_weather
from chap_core.climate_data.seasonal_forecasts import SeasonalForecast
from chap_core.datatypes import FullData, HealthData, HealthPopulationData, Samples, TimeSeriesArray, create_tsdataclass
from chap_core.rest_api.data_models import FetchRequest
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import convert_time_period_string

logger = logging.getLogger(__name__)
DISEASE_NAMES = ["disease", "diseases", "disease_cases"]


@dataclasses.dataclass
class DataValue:
    value: int
    orgUnit: str
    dataElement: str
    period: str


class WorkerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    is_test: bool = False
    failing_services: Tuple[str] = ()


def sample_dataset_to_prediction_response(predictions: DataSet[Samples], target_id: str) -> dict:
    summaries = DataSet({location: samples.summaries() for location, samples in predictions.items()})
    attrs = ["median", "quantile_high", "quantile_low"]
    data_values = predictions_to_datavalue(summaries, attribute_mapping=dict(zip(attrs, attrs)))
    json_body = [dataclasses.asdict(element) for element in data_values]
    response = {"diseaseId": target_id, "dataValues": json_body}
    return response


def dataset_to_datalist(dataset: DataSet[HealthData], target_id: str) -> DataList:
    element_list = [
        DataElement(pe=row.time_period.id, value=row.disease_cases, ou=location)
        for location, data in dataset.items()
        for row in data
    ]
    return DataList(dhis2Id=target_id, featureId="disease_cases", data=element_list)


def __clean_actual_cases(real_data: DataList) -> DataList:
    """Temporary function to clean time period names and fill nan valuse to a datalist of real cases"""
    df = pd.DataFrame([{"time_period": row.pe, "location": row.ou, "value": row.value} for row in real_data.data])
    print(df["time_period"])
    dataset = DataSet.from_pandas(df, TimeSeriesArray, fill_missing=True)
    return DataList(
        featureId=real_data.featureId,
        dhis2Id=real_data.dhis2Id,
        data=[
            DataElement(pe=row.time_period.id, ou=location, value=row.value if not np.isnan(row.value) else None)
            for location, ts_array in dataset.items()
            for row in ts_array
        ],
    )


def samples_to_evaluation_response(predictions_list, quantiles, real_data: DataList):
    evaluation_entries: List[EvaluationEntry] = []
    for predictions in predictions_list:
        first_period = predictions.period_range[0]
        for location, samples in predictions.items():
            calculated_quantiles = {q: np.quantile(samples.samples, q, axis=-1) for q in quantiles}
            for q, quantile in calculated_quantiles.items():
                for period, value in zip(predictions.period_range, quantile):
                    entry = EvaluationEntry(
                        orgUnit=location, period=period.id, quantile=q, value=value, splitPeriod=first_period.id
                    )
                    evaluation_entries.append(entry)
    real_data = __clean_actual_cases(real_data)
    return EvaluationResponse(actualCases=real_data, predictions=evaluation_entries)  # .model_dump()


def train_on_json_data(json_data: RequestV1, model_name, model_path, control=None):
    model_path = model_name
    json_data = PredictionRequest.model_validate_json(json_data)
    target_name = "diseases"
    target_id = get_target_id(json_data, target_name)
    train_data = dataset_from_request_v1(json_data)

    from chap_core.models.utils import get_model_from_directory_or_github_url

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
    json_data: RequestV1,
    target_name="diseases",
    usecwd_for_credentials=False,
    worker_config: WorkerConfig = WorkerConfig(),
) -> DataSet[FullData]:
    dataset = get_health_dataset(json_data)
    return harmonize_health_dataset(dataset, usecwd_for_credentials, worker_config=worker_config)


base_fetch_requests = (
    FetchRequest(feature_name="rainfall", data_source_name="total_precipitation"),
    FetchRequest(feature_name="mean_temperature", data_source_name="mean_2m_air_temperature"),
)


def harmonize_health_dataset(
    dataset,
    usecwd_for_credentials,
    fetch_requests: List[FetchRequest] = None,
    worker_config: WorkerConfig = WorkerConfig(),
):
    assert not fetch_requests, "Google earth engine no longer supported"
    return dataset


def get_health_dataset(json_data: PredictionRequest, dataclass=None, colnames=("ou", "pe")):
    if dataclass is None:
        dataclass = FullData if hasattr(json_data, "include_data") and json_data.include_data else HealthPopulationData

    target_name = get_target_name(json_data)
    translations = {target_name: "disease_cases"}
    population_feature = next(f for f in json_data.features if f.featureId == "population")
    if not len(population_feature.data):
        other_feature = next(f for f in json_data.features if f.featureId != "population")
        locations = {d.ou for d in other_feature.data}
        periods = {d.pe for d in other_feature.data}
        population_feature.data = [
            DataElement(pe=period, ou=location, value=10000000) for period in periods for location in locations
        ]

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
    """Get a dataset of potentially multiple data types from a RequestV1 object."""
    dataclass = create_tsdataclass([d.featureId for d in json_data.features])
    return get_health_dataset(json_data, dataclass)


def load_forecasts(data_path):
    climate_forecasts = SeasonalForecast()
    for file_name in os.listdir(data_path):
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


def v1_conversion(
    data_list: list[Union[DataElement, DataElementV2]], fill_missing=False, colnames=("ou", "pe")
) -> DataSet[TimeSeriesArray]:
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

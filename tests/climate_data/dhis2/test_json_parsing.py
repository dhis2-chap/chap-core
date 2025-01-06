import dataclasses
import json
import pytest

from chap_core.datatypes import SummaryStatistics
from chap_core.rest_api_src.worker_functions import predictions_to_datavalue
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange


@pytest.fixture()
def json_data(data_path):
    return json.load(open(data_path / "dhis_response.json"))


@pytest.fixture()
def population_data(data_path):
    return json.load(open(data_path / "population_Laos.json"))


@pytest.fixture()
def predictions():
    summary = SummaryStatistics(
        time_period=PeriodRange.from_strings(["2020-01", "2020-02"]),
        mean=[1.0, 2.0],
        std=[0.1, 0.2],
        median=[1.0, 2.0],
        min=[0.0, 0.0],
        max=[2.0, 4.0],
        quantile_low=[0.5, 1.5],
        quantile_high=[1.5, 2.5],
    )

    return DataSet({"loc1": summary})


def test_predictions_to_json_real(predictions):
    attrs = ["median", "quantile_high", "quantile_low"]
    data_values = predictions_to_datavalue(
        predictions, attribute_mapping=dict(zip(attrs, attrs))
    )
    json_body = [dataclasses.asdict(element) for element in data_values]
    assert len(json_body) == 6
    assert json_body[0]["dataElement"] == "median"

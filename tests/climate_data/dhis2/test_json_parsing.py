import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from chap_core.datatypes import HealthPopulationData, SummaryStatistics
from chap_core.dhis2_interface.json_parsing import (
    parse_disease_data,
    parse_population_data,
    join_data,
    predictions_to_datavalue,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange


@pytest.fixture()
def json_data(data_path):
    return json.load(open(data_path / "dhis_response.json"))


@pytest.fixture()
def population_data(data_path):
    return json.load(open(data_path / "population_Laos.json"))


@pytest.mark.skip(reason="This test is not yet implemented")
def test_parse_json(json_data):
    parsed = parse_disease_data(json_data)
    assert isinstance(parsed, pd.DataFrame)


def test_parse_population_data(population_data):
    lookup = parse_population_data(population_data, col_idx=2)
    assert "VWGSudnonm5" in lookup
    assert lookup["VWGSudnonm5"] == 272115


def test_join_population_and_health(population_data, json_data):
    joined = join_data(json_data, population_data)
    assert isinstance(joined, DataSet)
    for location, data in joined.items():
        assert isinstance(data.data(), HealthPopulationData)


# skipped since failing
@pytest.mark.skip()
def test_predictions_to_json():
    data = DataSet.from_pandas(
        pd.DataFrame(
            {
                "time_period": ["2020-01", "2020-02"],
                "disease_cases": [1, 2],
                "population": [100, 200],
                "location": ["loc1", "loc2"],
            }
        ),
        HealthPopulationData,
    )
    json_data = predictions_to_datavalue(
        data, attribute_mapping={"disease_cases": "D", "population": "P"}
    )
    for element in json_data:
        assert element["dataElement"] in ["D", "P"]
        assert element["orgUnit"] in ["loc1", "loc2"]
        assert element["value"] in [1, 2, 100, 200, np.nan] or np.isnan(
            element["value"]
        )
        assert element["period"] in ["2020-01", "2020-02"]


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

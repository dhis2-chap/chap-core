import pytest

from chap_core.dhis2_interface.json_parsing import (
    json_to_pandas,
    parse_disease_data,
)
from chap_core.time_period.date_util_wrapper import Month


@pytest.fixture
def json_data():
    data = {
        "rows": [
            ["ZMap7yJ5eCm", "jmIPBj66vD6", "202402", "28.8"],
            ["ZMap7yJ5eCm", "jmIPBj66vD6", "202403", "28.9"],
        ]
    }
    return data


def test_json_to_pandas(json_data):
    df = json_to_pandas(
        json_data, name_mapping={"time_period": 2, "disease_cases": 3, "location": 1}
    )
    print(all(df.time_period == ["2024-02", "2024-03"]))


def test_parse_disease_data(json_data):
    data = parse_disease_data(
        json_data, name_mapping={"time_period": 2, "disease_cases": 3, "location": 1}
    )
    first_period = list(data.data())[0].data().time_period[0]
    assert first_period == Month(2024, 2)  # , Month(2024, 3)]
    assert str(first_period.topandas()) == "2024-02"

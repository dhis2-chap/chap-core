import pandas as pd
import pytest

from chap_core.assessment.dataset_splitting import train_test_split_with_weather
from chap_core.datatypes import ClimateHealthTimeSeries
#from chap_core.external.models.jax_models.model_spec import NutsParams
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Month
from chap_core.datatypes import FullData

@pytest.fixture()
def data(data_path):
    file_name = (data_path / "hydro_met_subset").with_suffix(".csv")
    return DataSet.from_pandas(pd.read_csv(file_name), ClimateHealthTimeSeries)


@pytest.fixture()
def train_data(split_data):
    return split_data[0]


@pytest.fixture()
def split_data(data):
    return train_test_split_with_weather(data, Month(2013, 4))


@pytest.fixture()
def test_data(split_data):
    return split_data[1:]


@pytest.fixture
def full_train_data(train_data):
    return train_data.add_fields(FullData, population=lambda data: [100000] * len(data))

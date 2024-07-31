import pandas as pd
import pytest

from climate_health.assessment.dataset_splitting import train_test_split_with_weather
from climate_health.datatypes import ClimateHealthTimeSeries
from climate_health.external.models.jax_models.model_spec import NutsParams
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period import Month
from climate_health.datatypes import FullData

@pytest.fixture()
def blackjax():
    try:
        import blackjax
    except ImportError:
        pytest.skip("jax is not installed")
    return blackjax


@pytest.fixture()
def jax():
    try:
        import jax
    except ImportError:
        pytest.skip("jax is not installed")
    return jax


@pytest.fixture()
def pm():
    try:
        import pymc3 as pm
    except:
        pytest.skip("pymc3 is not installed")
    return pm


@pytest.fixture()
def fast_params():
    return NutsParams(n_samples=10, n_warmup=10)


@pytest.fixture()
def random_key(jax):
    return jax.random.PRNGKey(0)


@pytest.fixture()
def data(data_path):
    file_name = (data_path / 'hydro_met_subset').with_suffix('.csv')
    return SpatioTemporalDict.from_pandas(pd.read_csv(file_name), ClimateHealthTimeSeries)


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

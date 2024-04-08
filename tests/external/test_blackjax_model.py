from functools import partial

import pandas as pd
import pytest

from climate_health.assessment.dataset_splitting import train_test_split_with_weather
from climate_health.datatypes import ClimateHealthTimeSeries
from climate_health.external.models.jax_models.regression_model import RegressionModel, HierarchicalRegressionModel
from climate_health.external.models.jax_models.simple_ssm import SSM
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period import Month


@pytest.fixture()
def data(data_path):
    file_name = (data_path / 'hydro_met_subset').with_suffix('.csv')
    return SpatioTemporalDict.from_pandas(pd.read_csv(file_name), ClimateHealthTimeSeries)


@pytest.fixture()
def train_data(split_data):
    return split_data[0]


@pytest.fixture()
def split_data(data):
    return train_test_split_with_weather(data, Month(2012, 8))


@pytest.fixture()
def test_data(split_data):
    return split_data[1:]


def test_blackjax_model_train(blackjax, jax, train_data, model):
    model.train(train_data)


def test_hierarchical_model_train(blackjax, jax, train_data, hierarchical_model):
    hierarchical_model.train(train_data)


@pytest.fixture()
def init_values(jax):
    jnp = jax.numpy
    return {'beta_temp': 0.1, 'beta_lag': 0.1, 'beta_season': jnp.full(12, 0.1)}


@pytest.fixture()
def model(simple_priors, init_values):
    return RegressionModel(simple_priors, init_values, num_warmup=100, num_samples=100)

@pytest.fixture()
def hierarchical_model(simple_priors, init_values):
    return HierarchicalRegressionModel(num_warmup=100, num_samples=100)


@pytest.fixture()
def simple_priors(jax):
    season_prior = partial(jax.scipy.stats.norm.logpdf, 0, 1)
    beta_prior = partial(jax.scipy.stats.norm.logpdf, 0, 1)
    priors = {'beta_temp': beta_prior,
              'beta_lag': beta_prior,
              'beta_season': season_prior}
    return priors


def test_blackjax_model_predict(model, train_data, test_data):
    truth, future_data = test_data
    model.train(train_data)
    model.predict(future_data)

def test_hierarchical_model_predict(hierarchical_model, train_data, test_data):
    truth, future_data = test_data
    hierarchical_model.train(train_data)
    hierarchical_model.predict(future_data)

def test_ssm_train(train_data, test_data):
    model = SSM()
    model.train(train_data)

def test_ssm_test(train_data, test_data):
    truth, future_data = test_data
    model = SSM()
    model.train(train_data)
    model.predict(future_data)

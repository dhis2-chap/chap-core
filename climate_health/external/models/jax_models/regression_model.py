from functools import partial
from typing import Optional

import numpy as np
from jax.scipy import stats
from jax.random import PRNGKey
import jax.numpy as jnp

from climate_health.dataset import ClimateData
from climate_health.datatypes import HealthData
from .hmc import sample
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
import jax


def simple_priors():
    season_prior = partial(jax.scipy.stats.norm.logpdf, 0, 1)
    beta_prior = partial(jax.scipy.stats.norm.logpdf, 0, 1)
    priors = {'beta_temp': beta_prior,
              'beta_lag': beta_prior,
              'beta_season': season_prior}
    return priors


def init_values():
    jnp = jax.numpy
    return {'beta_temp': 0.1, 'beta_lag': 0.1, 'beta_season': jnp.full(12, 0.1)}


class RegressionModel:

    def __init__(self, priors: Optional[dict] = None, initial_params=None, num_warmup=100, num_samples=300):

        self._param_samples = None
        if priors is None:
            priors = simple_priors()
        if initial_params is None:
            initial_params = init_values()
        self._priors = priors
        self._initial_params = initial_params
        self._num_warmup = num_warmup
        self._num_samples = num_samples

    def _default_prior(self):
        return simple_priors()

    def _default_init_values(self):
        return init_values()

    def _get_rate(self, params, data, season, disease_cases, location=None):
        temp_part = params['beta_temp'] * data.mean_temperature
        lag_part = params['beta_lag'] * jnp.log(disease_cases)
        season_part = params['beta_season'][season]
        return jnp.exp(temp_part + lag_part + season_part)

    def train(self, st_data: SpatioTemporalDict):

        tmp_data_list = [location_data.data() for location_data in st_data.data()]
        data_list = []
        for data in tmp_data_list:
            nans = np.isnan(data.disease_cases)
            first_non_nan = np.where(~nans)[0][0]
            data_list.append(data[first_non_nan:])
        seasons = [np.array([period.month for period in data.time_period]) for data in data_list]

        def logpdf(params):
            prior_pdf = [jnp.sum(self._priors[name](params[name])) for name in self._priors]
            for i, (data, season) in enumerate(zip(data_list, seasons)):
                rate = self._get_rate(params, data[1:], season[1:],
                                      data.disease_cases[:-1], i)  # temp_part + lag_part + season_part
                obs_pdf = stats.poisson.logpmf(
                    data.disease_cases[1:], rate).sum()
            return sum(prior_pdf) + obs_pdf

        logpdf(self._initial_params)

        self._param_samples = sample(logpdf, PRNGKey(348), self._initial_params,
                                     num_samples=self._num_samples, num_warmup=self._num_warmup)
        self._state = {location: data.data().disease_cases[-1:] for location, data in st_data.items()}

    def predict(self, st_data: SpatioTemporalDict[ClimateData]):
        params = {name: self._param_samples[name].mean(axis=0) for name in self._priors}
        location_names = st_data.locations()
        data_list = [location_data.data()[:1] for location_data in st_data.data()]
        seasons = [np.array([period.month for period in data.time_period]) for data in data_list]
        predictions = []
        for i, (location, data, season) in enumerate(zip(location_names, data_list, seasons)):
            rate = self._get_rate(params, data, season, self._state[location], i)
            predictions.append(HealthData(data.time_period, rate))
        print(predictions)
        return SpatioTemporalDict(
            {location: prediction for location, prediction in zip(st_data.locations(), predictions)})


class HierarchicalRegressionModel(RegressionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._priors['location_offset']= partial(jax.scipy.stats.norm.logpdf, 0, 1)

    def train(self, st_data: SpatioTemporalDict):
        self._initial_params['location_offset'] = jnp.zeros(len(st_data.locations()))
        super().train(st_data)

    def _get_rate(self, params, data, season, disease_cases, location=None):
        temp_part = params['beta_temp'] * data.mean_temperature
        lag_part = params['beta_lag'] * jnp.log(disease_cases)
        season_part = params['beta_season'][season]
        location_part = params['location_offset'][location]
        return jnp.exp(temp_part + lag_part + season_part+location_part)


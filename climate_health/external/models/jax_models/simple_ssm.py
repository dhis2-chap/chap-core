from collections import defaultdict
from functools import partial

import numpy as np

from climate_health.time_period.date_util_wrapper import TimeDelta
from .util import extract_last, extract_sample, array_tree_length
from climate_health.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData, SummaryStatistics
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from .hmc import sample
from .jax import jax, PRNGKey, stats, jnp
from .regression_model import remove_nans
import logging

logger = logging.getLogger(__name__)


def get_summary(time_period, samples):
    statistics = (np.mean(samples, axis=0),
                  np.median(samples, axis=0),
                  np.std(samples, axis=0),
                  np.min(samples, axis=0),
                  np.max(samples, axis=0),
                  np.percentile(samples, 2.5, axis=0),
                  np.percentile(samples, 97.5, axis=0))

    return SummaryStatistics(time_period, *(np.atleast_1d(d) for d in statistics))


class NaiveSSM:
    n_warmup = 200

    def __init__(self):
        self._last_temperature = None
        self._key = PRNGKey(124)
        self._sampled_params = None
        self._initial_params = {'infected_decay': 0.9, 'alpha': 0.1}
        self._prior = self._get_priors(self._initial_params.keys())

    def _get_priors(self, param_names):
        return {name: partial(stats.norm.logpdf, loc=0.1, scale=1) for name in param_names}

    def _log_infected_dist(self, params, prev_log_infected, mean_temp, scale=1.0):
        mu = prev_log_infected * params['infected_decay'] + self._temperature_effect(mean_temp, params)
        return partial(stats.norm.logpdf, loc=mu, scale=scale), lambda key, shape: jax.random.normal(key,
                                                                                                     shape) * scale + mu

    def _temperature_effect(self, mean_temp, params):
        return params['alpha']

    def train(self, data: SpatioTemporalDict[ClimateHealthTimeSeries]):
        orig_data = data
        logger.info(f"Training model with {len(data.locations())} locations")
        data = {name: remove_nans(data.data())
                for name, data in data.items()}
        init_params = self._initial_params
        init_params['log_infected'] = {location: jnp.log(data.disease_cases + 1)
                                       for location, data in data.items()}
        init_params['log_observation_rate'] = {location: jnp.log(0.1) for location in data.keys()}
        prior_pdf_func = lambda params: sum([prior(params[name]) for name, prior in self._prior.items()])

        def obs_pdf_func(params, local_data, location):
            rate = jnp.exp(params['log_infected'][location] + params['log_observation_rate'][location])
            obs_p = stats.poisson.logpmf(local_data.disease_cases, rate).sum()
            return obs_p

        def infected_pdf_func(params, local_data, location):
            infected_dist, _ = self._log_infected_dist(params, params['log_infected'][location][:-1],
                                                       local_data.mean_temperature[:-1])
            return infected_dist(params['log_infected'][location][1:]).sum()

        def logpdf(params):
            obs_pdf = []
            state_pdf = []
            prior_pdf = prior_pdf_func(params)
            for location, local_data in data.items():
                obs_p = obs_pdf_func(params, local_data, location)
                obs_pdf.append(obs_p)
                infected_pdf = infected_pdf_func(params, local_data, location)
                state_pdf.append(infected_pdf)

            return sum(obs_pdf) + sum(state_pdf) + prior_pdf

        first_pdf = logpdf(init_params)
        if jnp.isnan(first_pdf):
            logger.error(prior_pdf_func(init_params))
            logger.error([obs_pdf_func(init_params, local_data, location) for location, local_data in data.items()])
            logger.error(
                [infected_pdf_func(init_params, local_data, location) for location, local_data in data.items()])
        assert not jnp.isnan(first_pdf)
        jax.grad(logpdf)(init_params)
        self._sampled_params = sample(logpdf, PRNGKey(0), init_params, 100, self.n_warmup)
        self._last_temperature = {location:  data.data().mean_temperature[-1]
                                  for location, data in orig_data.items()}

        last_params = extract_last(self._sampled_params)
        last_pdf = logpdf(last_params)
        assert not jnp.isnan(last_pdf)

    def predict(self, data: SpatioTemporalDict[ClimateData]):
        last_params = extract_last(self._sampled_params)
        predictions = {}
        for location, local_data in data.items():
            time_period = data.get_location(location).data().time_period[:1]
            rate = self._get_rate(last_params, location, self._last_temperature[location])
            predictions[location] = HealthData(time_period, rate)

        return SpatioTemporalDict(predictions)

    def _get_rate(self, last_params, location, temperature):
        self._key, key = jax.random.split(self._key)
        log_infected = last_params['log_infected'][location]
        _, sampler = self._log_infected_dist(last_params, log_infected, temperature)
        log_infected = sampler(key, (1,))
        log_observation_rate = last_params['log_observation_rate'][location]
        rate = jnp.exp(log_infected[-1:] + log_observation_rate)
        return rate

    def _get_rates_forwards(self, last_params, location, climate_data, n_periods):
        self._key, key = jax.random.split(self._key)
        log_infected = last_params['log_infected'][location][-1]
        state = log_infected
        temp = self._last_temperature[location]
        log_observation_rate = last_params['log_observation_rate'][location]
        states=[]
        for i, key in enumerate(jax.random.split(key, n_periods)):
            _, sampler = self._log_infected_dist(last_params, state, temp)
            state = sampler(key, ())
            temp = climate_data.mean_temperature[i]
            states.append(state)
        return np.exp(np.array(states)+log_observation_rate)

    def prediction_summary(self, data: SpatioTemporalDict, num_samples: int = 100) -> SpatioTemporalDict[
        SummaryStatistics]:
        self._key, param_key, sample_key = jax.random.split(self._key, 3)
        n_sampled_params = array_tree_length(self._sampled_params)
        param_idxs = jax.random.randint(param_key, (num_samples,), 0, n_sampled_params)
        samples = defaultdict(list)
        for i, key in zip(param_idxs, jax.random.split(sample_key, num_samples)):
            param = extract_sample(i, self._sampled_params)
            for location, local_data in data.items():
                rate = self._get_rate(param, location, self._last_temperature[location])
                samples[location].append(jax.random.poisson(key, rate))
        time_period = next(iter(data.data())).data().time_period[:1]
        summaries = {k: get_summary(time_period, s) for k, s in samples.items()}
        return SpatioTemporalDict(summaries)

    def forecast(self, data: SpatioTemporalDict[ClimateData], num_samples: int = 100,
                 forecast_delta=TimeDelta) -> SpatioTemporalDict:
        self._key, param_key, sample_key = jax.random.split(self._key, 3)
        time_period = next(iter(data.data())).data().time_period
        n_periods = forecast_delta // time_period.delta
        time_period = time_period[:n_periods]
        n_sampled_params = array_tree_length(self._sampled_params)
        param_idxs = jax.random.randint(param_key, (num_samples,), 0, n_sampled_params)
        samples = defaultdict(list)

        for i, key in zip(param_idxs, jax.random.split(sample_key, num_samples)):
            param = extract_sample(i, self._sampled_params)
            for location, local_data in data.items():
                rates = self._get_rates_forwards(param, location, local_data.data(),  n_periods)
                samples[location].append(jax.random.poisson(key, rates))
        summaries = {k: get_summary(time_period, s)
                     for k, s in samples.items()}
        return SpatioTemporalDict(summaries)


class SeasonalSSM(NaiveSSM):
    def __init__(self):
        super().__init__()
        self._initial_params['seasonal_effect'] = jnp.zeros(12)
        self._prior['seasonal_effect'] = partial(stats.norm.logpdf, loc=0, scale=1)

    def _temperature_effect(self, mean_temp, params):
        return params['alpha'] + jnp.sum(params['seasonal_effect'][mean_temp.month - 1])


class SSM(NaiveSSM):
    def __init__(self):
        self._key = PRNGKey(124)
        self._initial_params = {'infected_decay': 0.9, 'beta_temp': 0.1}
        self._prior = self._get_priors(self._initial_params.keys())

    def _temperature_effect(self, mean_temp, params):
        return params['beta_temp'] * mean_temp


class SSMWithLinearEffect(SSM):
    """ SSM where the temperature effect is modelled as a sigmoid function of temperature. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_params['alpha_temp'] = 0.1
        self._prior['alpha_temp'] = partial(stats.norm.logpdf, loc=0.1, scale=0.1)

    def _temperature_effect(self, mean_temp, params):
        return params['beta_temp'] * mean_temp + params['alpha_temp']


class SSMWithSigmoidEffect(SSM):
    ''' SSM where the temperature effect is modelled as a sigmoid function of temperature.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        param_names = ['a_temp', 'b_temp', 'c_temp']
        self._initial_params.update({name: 0.1 for name in param_names})
        self._prior.update({name: partial(stats.norm.logpdf, loc=0.1, scale=1) for name in param_names})

    def _temperature_effect(self, mean_temp, params):
        return jax.scipy.special.expit(params['beta_temp'] * mean_temp + params['a_temp']) * params['b_temp'] + params[
            'c_temp']


class SSMWithScaleParam(SSM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_params['log_scale'] = 0.1
        self._prior['log_scale'] = partial(stats.norm.logpdf, loc=0.1, scale=0.1)

    def _log_infected_dist(self, params, prev_log_infected, mean_temp, scale=1.0):
        return super()._log_infected_dist(params, prev_log_infected, mean_temp, scale=jnp.exp(params['log_scale']))

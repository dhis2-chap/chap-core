from dataclasses import dataclass
from typing import Protocol, Any

import numpy as np

from climate_health.datatypes import ClimateHealthTimeSeries, HealthData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from .hmc import sample
from .jax import jax, stats, jnp, PRNGKey
from .regression_model import remove_nans
from .util import extract_last, index_tree

expit = jax.scipy.special.expit
logit = jax.scipy.special.logit

from climate_health.predictor.poisson import Poisson
import logging

logger = logging.getLogger(__name__)


class IsDistribution(Protocol):
    def sample(self, key, n: int) -> Any:
        ...

    def log_prob(self, x: Any) -> Any:
        ...


@dataclass
class Normal:
    mu: float
    sigma: float

    def sample(self, key, shape: int) -> Any:
        return jax.random.normal(key, shape) * self.sigma + self.mu

    def log_prob(self, x: Any) -> Any:
        return stats.norm.logpdf(x, loc=self.mu, scale=self.sigma)


@dataclass
class Poisson:
    rate: float

    def sample(self, key, shape: int) -> Any:
        return jax.random.poisson(key, self.rate, shape)

    def log_prob(self, x: Any) -> Any:
        return stats.poisson.logpmf(x, self.rate)


class IsSSMSpec(Protocol):
    global_params: list[str]
    state_params: list[str]

    @staticmethod
    def observation_distribution(self, params: dict[str, Any]) -> IsDistribution:
        ...

    def state_distribution(self, previous_state: dict, params: dict[str, Any], observed: dict) -> IsDistribution:
        ...



class NaiveSSM:
    global_params = ['beta_temp']
    state_params = ['logit_infected']
    location_params = ['logit_infected_decay', 'log_observation_rate']
    predictors = ['mean_temperature']

    def observation_distribution(self, state: dict[str, Any], params: dict[str, Any]) -> IsDistribution:
        return Poisson(jnp.exp(state['logit_infected'] + params['log_observation_rate']))

    def _temperature_effect(self, temperature: float, params: dict[str, Any]) -> float:
        return params['beta_temp'] * temperature

    def state_distribution(self, previous_state: dict, params: dict[str, Any], observed: dict) -> IsDistribution:
        mu = previous_state['logit_infected'] * expit(params['logit_infected_decay']) + self._temperature_effect(
            observed['mean_temperature'], params)
        return DictDist(Normal(mu, 1.0))


class DictDist:
    def __init__(self, dist):
        self.dist = dist

    def __getitem__(self, item):
        return self.dist

    def log_prob(self, x):
        return sum(self.dist.log_prob(value) for value in x.values())


class IsSSMForecaster(Protocol):

    def __init__(self, model_spec: IsSSMSpec, training_params: dict[str, Any] = None):
        ...

    def train(self, data: dict[str, Any]):
        ...

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        ...

    def sample(self, key, n: int) -> dict[str, Any]:
        ...

    def forecast(self, data: dict[str, Any], n: int, forecast_delta: int) -> dict[str, Any]:
        ...


@dataclass
class NutsParams:
    n_samples: int = 100
    n_warmup: int = 100


class SSMForecasterNuts:
    def __init__(self, model_spec: IsSSMSpec, training_params: dict[str, Any] = None):
        self._model_spec = model_spec
        if training_params is None:
            training_params = NutsParams()
        self._training_params = training_params

    def _get_local_params(self, params, location):
        return {name: params[name] for name in self._model_spec.global_params} | {param_name: params[param_name][location] for param_name in self._model_spec.location_params}


    def train(self, data: SpatioTemporalDict[ClimateHealthTimeSeries]):
        orig_data = data
        logger.info(f"Training model {self._model_spec} with {len(data.locations())} locations")
        data = {name: remove_nans(data.data())
                for name, data in data.items()}
        init_params = self._get_init_params(data)
        prior = self._get_prior()
        prior_pdf_func = prior.log_prob

        def obs_pdf_func(states, local_params, local_data):
            return self._model_spec.observation_distribution(states, local_params).log_prob(
                local_data.disease_cases).sum()

        def infected_pdf_func(states, params, local_data):
            previous_state = {param_name: states[param_name][:-1] for param_name in self._model_spec.state_params}
            new_state = {param_name: states[param_name][1:] for param_name in self._model_spec.state_params}
            dist = self._model_spec.state_distribution(previous_state, params,
                                                       {name: getattr(local_data, name)[:-1] for name in
                                                        self._model_spec.predictors})

            return dist.log_prob(new_state).sum()


        def logpdf(params):
            obs_pdf = []
            state_pdf = []
            prior_pdf = prior_pdf_func(params)
            for location, local_data in data.items():
                local_params = self._get_local_params(params, location)
                states = self._get_state_params(params, location)
                obs_p = obs_pdf_func(states, local_params, local_data)
                obs_pdf.append(obs_p)
                infected_pdf = infected_pdf_func(states, local_params, local_data)
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
        self._sampled_params = sample(logpdf, PRNGKey(0), init_params, self._training_params.n_samples,
                                      self._training_params.n_warmup)
        self._last_temperature = {location: data.data().mean_temperature[-1]
                                  for location, data in orig_data.items()}

        last_params = extract_last(self._sampled_params)
        last_pdf = logpdf(last_params)
        assert not jnp.isnan(last_pdf)

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        last_params = extract_last(self._sampled_params)
        predictions = {}
        for location, local_data in data.items():
            location_params = self._get_local_params(last_params, location)
            new_state = self._model_spec.state_distribution(index_tree(self._get_state_params(last_params, location), -1),
                                                            location_params,
                                                            {'mean_temperature': self._last_temperature[location]})['logit_infected'].mu
            rate = self._model_spec.observation_distribution(new_state, location_params).rate
            time_period = data.get_location(location).data().time_period[:1]
            predictions[location] = HealthData(time_period, np.atleast_1d(rate))

        return SpatioTemporalDict(predictions)


    def sample(self, key, n: int) -> dict[str, Any]:
        ...

    def forecast(self, data: dict[str, Any], n: int, forecast_delta: int) -> dict[str, Any]:
        ...

    def _get_init_params(self, data):
        init_params = {param_name: 0.1 for param_name in self._model_spec.global_params}
        for name in self._model_spec.state_params:
            init_params[name] = {location: jnp.full(len(data), 0.1) for location, data in data.items()}
        for name in self._model_spec.location_params:
            init_params[name] = {location: jnp.log(0.1) for location in data.keys()}
        return init_params

    def _get_prior(self):
        log_pdf = lambda params: sum(
            Normal(0, 1).log_prob(params[name]) for name in self._model_spec.global_params) + sum(
            Normal(0, 1).log_prob(value) for name in self._model_spec.location_params for value in
            params[name].values())
        return PriorDistribution(log_pdf)

    def _get_state_params(self, params, location):
        return {param_name: params[param_name][location] for param_name in self._model_spec.state_params}


class PriorDistribution:
    def __init__(self, log_prob_func):
        self._log_prob_func = log_prob_func

    def log_prob(self, x: dict) -> Any:
        return self._log_prob_func(x)

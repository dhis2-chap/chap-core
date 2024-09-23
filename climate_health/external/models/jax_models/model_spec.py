import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Protocol, Any, Optional, Sequence, Callable

import numpy as np
import scipy

from climate_health.datatypes import ClimateHealthTimeSeries, HealthData, SummaryStatistics, ClimateData
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet
from climate_health.time_period.date_util_wrapper import delta_month, TimeDelta, TimePeriod, PeriodRange
from .hmc import sample
from .jax import jax, stats, jnp, PRNGKey, expit, logit
from .regression_model import remove_nans
from .simple_ssm import get_summary
from .util import extract_last, index_tree, array_tree_length, extract_sample

import logging

logger = logging.getLogger(__name__)


class IsDistribution(Protocol):
    def sample(self, key, shape: Optional[tuple] = None) -> Any:
        ...

    def log_prob(self, x: Any) -> Any:
        ...


distributionclass = partial(dataclass, frozen=True)


@distributionclass
class Normal:
    mu: float
    sigma: float
    ndim: int = 0

    def sample(self, key, shape: Sequence[int] = ()) -> Any:
        assert shape==()
        if hasattr(self.mu, 'shape'):
            shape= self.mu.shape
        return jax.random.normal(key, shape) * self.sigma + self.mu

    def log_prob(self, x: Any) -> Any:
        if self.ndim > 0:
            assert self.ndim == 1
            return stats.norm.logpdf(x, loc=self.mu, scale=self.sigma).sum()
        return stats.norm.logpdf(x, loc=self.mu, scale=self.sigma)


@distributionclass
class NegativeBinomial:
    k: float
    p: float

    def log_prob(self, x: Any) -> Any:
        return stats.nbinom.logpmf(x, self.k, self.p)

    def mean(self):
        return self.k * (1 - self.p) / self.p

'''
mu = k*(1-p)/p
k = mu/(1-p)*p
k = mu/phi
phi = p/(1-p)
(1-p)*phi = p
phi-p*phi = p
phi = p(1+phi)
p = phi/(1+phi)
'''

@distributionclass
class NegativeBinomial2:
    mu: float
    alpha: float

    def log_prob(self, x: Any) -> Any:
        log_gamma = jax.scipy.special.gammaln(
            x + 1/self.alpha) - jax.scipy.special.gammaln(1/self.alpha) - jax.scipy.special.gammaln(x + 1)
        first_term = 1/self.alpha * jnp.log(1 + self.alpha * self.mu)
        second_term = x * (jnp.log(self.mu)+jnp.log(self.alpha)-jnp.log(1+self.alpha*self.mu))
        return log_gamma - first_term + second_term

    def mean(self):
        return self.mu

    def sigma(self):
        return self.mu + self.mu**2*self.alpha

    def p(self):
        return self.alpha/(1+self.alpha* self.mu)

    def n(self):
        return 1/self.alpha

@distributionclass
class NegativeBinomial3():
    def __init__(self, total_count, logits=None):
        self.total_count = total_count
        self.logits = logits
        self.probs = jax.nn.sigmoid(logits)
    @property
    def mean(self):
        return self.total_count * jnp.exp(self.logits)

    @property
    def mode(self):
        return ((self.total_count - 1) * self.logits.exp()).floor().clamp(min=0.0)

    @property
    def variance(self):
        return self.mean / jax.nn.sigmoid(-self.logits)

    def log_prob(self, value):
        log_unnormalized_prob = self.total_count * jax.nn.log_sigmoid(
            -self.logits
        ) + value * jax.nn.log_sigmoid(self.logits)

        log_normalization = (
            -jax.lax.lgamma(self.total_count + value)
            + jax.lax.lgamma(1.0 + value)
            + jax.lax.lgamma(self.total_count)
        )
        # The case self.total_count == 0 and value == 0 has probability 1 but
        # lgamma(0) is infinite. Handle this case separately using a function
        # that does not modify tensors in place to allow Jit compilation.
        #log_normalization = log_normalization.masked_fill(
        #    self.total_count + value == 0.0, 0.0
        #)

        return log_unnormalized_prob - log_normalization

    def cdf(self, value):
        result = self.scipy_nbinom.cdf(value.detach().cpu().numpy())
        return result

    def icdf(self, value):
        result = self.scipy_nbinom.ppf(value)
        return result


    @property
    def scipy_nbinom(self):
        return scipy.stats.nbinom(
            n=self.total_count,
            p=1.0 - self.probs)





@distributionclass
class Poisson:
    rate: float

    def sample(self, key, shape: tuple = ()) -> Any:
        assert shape==()
        if hasattr(self.rate, 'shape'):
            shape= self.rate.shape
        return jax.random.poisson(key, self.rate, shape)

    def log_prob(self, x: Any) -> Any:
        return stats.poisson.logpmf(x, self.rate)


@dataclass(frozen=True)
class Exponential:
    beta: float

    def sample(self, key, shape: Optional[tuple] = ()) -> Any:
        return jax.random.exponential(key, shape) * self.beta

    def log_prob(self, x: float):
        return stats.expon.logpdf(x, scale=self.beta)


@dataclass(frozen=True)
class LogNormal:
    mu: float
    sigma: float

    def sample(self, key, shape: Optional[tuple] = ()) -> Any:
        return jax.random.lognormal(key, self.sigma, shape) * np.exp(self.mu)

    def log_prob(self, value: float):
        np = jnp
        return -np.log(value) - np.log(self.sigma) - 0.5 * np.log(2 * np.pi) + np.log(value - self.mu) ** 2 / (
                    2 * self.sigma ** 2)


class PoissonSkipNaN(Poisson):
    def log_prob(self, x: Any) -> Any:
        nans = jnp.isnan(x)
        rate = jnp.where(nans, 0, self.rate)
        masked = jnp.where(nans, rate, x)
        res = jnp.where(nans, 0, stats.poisson.logpmf(masked, rate))
        return res

def skip_nan_distribution(dist: IsDistribution):
    class SkipNaN(dist):
        def log_prob(self, x: Any) -> Any:
            nans = jnp.isnan(x)
            masked = jnp.where(nans, 0, x)
            res = jnp.where(nans, 0, super().log_prob(masked))
            return res

    return SkipNaN


class NormalSkipNaN(Normal):
    ...


class IsSSMSpec(Protocol):
    global_params: list[str]
    state_params: list[str]

    @staticmethod
    def observation_distribution(self, params: dict[str, Any], predictors: dict[str, Any] = None) -> IsDistribution:
        ...

    def state_distribution(self, previous_state: dict, params: dict[str, Any], observed: dict) -> IsDistribution:
        ...


class DictDist:
    def __init__(self, dist_dict):
        self.dist_dict = dist_dict

    def __getitem__(self, item):
        return self.dist_dict[item]

    def log_prob(self, x):
        return sum(self.dist_dict[name].log_prob(x[name]) for name in self.dist_dict.keys())

    def sample(self, key, shape=()):
        return {name: dist.sample(key, shape) for name, dist in self.dist_dict.items()}


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
    def __init__(self, model_spec: IsSSMSpec, training_params: dict[str, Any] = None, key: "PRNGKey" = None):
        self._sampled_params = None
        self._last_predictors = None
        if key is None:
            key = PRNGKey(0)
        self._model_spec = model_spec
        if training_params is None:
            training_params = NutsParams()
        self._training_params = training_params
        self._key = key

    def save(self, filename: str):
        ''' Pickle everything neeeded to recreate the model.'''
        with open(filename, 'wb') as f:
            pickle.dump((self._sampled_params, self._last_predictors, self._model_spec), f)

    @classmethod
    def load(cls, filename: str):
        ''' Load a model from a pickled file.'''
        with open(filename, 'rb') as f:
            sampled_params, last_predictors, model_spec = pickle.load(f)
        model = cls(model_spec)
        model._sampled_params = sampled_params
        model._last_predictors = last_predictors
        return model

    def _get_local_params(self, params, location):
        globals = {name: params[name] for name in self._model_spec.global_params}
        locals = {param_name: params[param_name][location] for param_name in self._model_spec.location_params}
        return globals | locals

    def train(self, data: DataSet[ClimateHealthTimeSeries]):
        orig_data = data
        logger.info(f"Training model {self._model_spec} with {len(data.locations())} locations")
        data = {name: remove_nans(data.data())
                for name, data in data.items()}
        init_params = self._get_init_params(data)
        prior = self._get_prior()
        prior_pdf_func = prior.log_prob

        def obs_pdf_func(states, local_params, local_data):
            return self._model_spec.observation_distribution(
                states, local_params,
                {name: getattr(local_data, name) for name in self._model_spec.predictors}).log_prob(
                local_data.disease_cases).sum()

        def infected_pdf_func(states, params, local_data):
            previous_state = {param_name: states[param_name][:-1]
                              for param_name in self._model_spec.state_params}
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
            logger.error([obs_pdf_func(self._get_state_params(init_params, location),
                                       self._get_local_params(init_params, location), local_data)
                          for location, local_data in data.items()])
            logger.error(
                [infected_pdf_func(self._get_state_params(init_params, location),
                                   self._get_local_params(init_params, location),
                                   local_data) for location, local_data in data.items()])
        assert not jnp.isnan(first_pdf)
        jax.grad(logpdf)(init_params)
        self._sampled_params = sample(logpdf, PRNGKey(0), init_params, self._training_params.n_samples,
                                      self._training_params.n_warmup)
        self._last_predictors = {
            location: {name: getattr(data.data(), name)[-1] for name in self._model_spec.predictors}
            for location, data in orig_data.items()}
        last_params = extract_last(self._sampled_params)
        last_pdf = logpdf(last_params)
        assert not jnp.isnan(last_pdf)

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        last_params = extract_last(self._sampled_params)
        predictions = {}
        for location, local_data in data.items():
            location_params = self._get_local_params(last_params, location)
            new_state = \
                self._model_spec.state_distribution(index_tree(self._get_state_params(last_params, location), -1),
                                                    location_params,
                                                    self._last_predictors[location])['logit_infected'].mu
            rate = self._model_spec.observation_distribution({'logit_infected': new_state}, location_params).rate
            time_period = data.get_location(location).data().time_period[:1]
            predictions[location] = HealthData(time_period, np.atleast_1d(rate))

        return DataSet(predictions)

    def prediction_summary(self, data: DataSet = None, num_samples: int = 100) -> DataSet[
        SummaryStatistics]:
        if isinstance(data, TimePeriod):
            time_period = PeriodRange.from_time_periods(data, data)
            locations = list(self._last_predictors.keys())
        elif isinstance(data, DataSet):
            locations = list(data.locations())
            time_period = next(iter(data.data())).data().time_period[:1]
        self._key, param_key, sample_key = jax.random.split(self._key, 3)
        n_sampled_params = array_tree_length(self._sampled_params)
        param_idxs = jax.random.randint(param_key, (num_samples,), 0, n_sampled_params)
        samples = defaultdict(list)
        # locations = list(self._last_predictors.keys()) if data is None else data.locations()
        for i, key in zip(param_idxs, jax.random.split(sample_key, num_samples)):
            state_key, obs_key = jax.random.split(key)
            param = extract_sample(i, self._sampled_params)
            for location in locations:
                location_params = self._get_local_params(param, location)
                last_state = index_tree(self._get_state_params(param, location), -1)
                new_state = self._model_spec.state_distribution(
                    last_state, location_params,
                    self._last_predictors[location]).sample(state_key)
                samples[location].append(
                    self._model_spec.observation_distribution(new_state, location_params,
                                                              self._last_predictors[location]).sample(obs_key))
        # time_period = next(iter(data.data())).data().time_period[:1]
        summaries = {k: get_summary(time_period, s) for k, s in samples.items()}
        return DataSet(summaries)

    def sample(self, key, n: int) -> dict[str, Any]:
        ...

    def forecast(self, data: DataSet[ClimateData], num_samples: int = 100,
                 forecast_delta: TimeDelta = 3 * delta_month) -> DataSet:
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
                rates = self._get_rates_forwards(param, location, local_data.data(), n_periods)
                samples[location].append(jax.random.poisson(key, rates))
        summaries = {k: get_summary(time_period, s)
                     for k, s in samples.items()}
        return DataSet(summaries)

    def _get_init_params(self, data):
        init_params = {param_name: 0.1 for param_name in self._model_spec.global_params}
        for name in self._model_spec.state_params:
            init_params[name] = {location: jnp.full(len(data), 0.1) for location, data in data.items()}
        for name in self._model_spec.location_params:
            init_params[name] = {location: jnp.log(0.1) for location in data.keys()}
        return init_params

    def _get_prior(self):
        log_pdf = lambda params: sum(
            Normal(0, 10).log_prob(params[name]) for name in self._model_spec.global_params) + sum(
            Normal(0, 10).log_prob(value) for name in self._model_spec.location_params for value in
            params[name].values())
        return PriorDistribution(log_pdf)

    def _get_state_params(self, params, location):
        return {param_name: params[param_name][location] for param_name in self._model_spec.state_params}


class PriorDistribution:
    def __init__(self, log_prob_func):
        self._log_prob_func = log_prob_func

    def log_prob(self, x: dict) -> Any:
        return self._log_prob_func(x)

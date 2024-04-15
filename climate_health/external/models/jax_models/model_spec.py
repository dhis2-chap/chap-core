from dataclasses import dataclass
from typing import Protocol, Any
from .jax import jax, stats, jnp
expit = jax.scipy.special.expit
logit = jax.scipy.special.logit

from climate_health.predictor.poisson import Poisson


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
        return stats.norm.logpdf(x, loc = self.mu, scale = self.scale)

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

    def state_distribtion(self, params: dict[str, Any]) -> IsDistribution:
        ...


class NaiveSSM:
    params = ['logit_infected_decay', 'beta_temp']
    state_params = ['logit_infected']

    def observation_distribution(self, params: dict[str, Any]) -> IsDistribution:
        return Poisson(jnp.exp(params['logit_infected']))

    def state_distribution(self, previous_state: dict, params: dict[str, Any], observed: dict) -> IsDistribution:
        mu = previous_state['logit_infected'] * expit(params['logit_infected_decay']) + self._temperature_effect(
            observed['mean_temp'], params)
        return Normal(mu, 1.0)



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




class SSMForecasterNuts:
    def __init__(self, model_spec: IsSSMSpec, training_params: dict[str, Any] = None):
        self._model_spec = model_spec
        self._training_params = training_params

    def train(self, data: dict[str, Any]):
        ...

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        ...

    def sample(self, key, n: int) -> dict[str, Any]:
        ...

    def forecast(self, data: dict[str, Any], n: int, forecast_delta: int) -> dict[str, Any]:
        ...

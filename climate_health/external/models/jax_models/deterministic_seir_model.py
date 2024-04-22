import dataclasses
from functools import partial
from typing import TypeVar, Generic, Callable
from bionumpy import bnpdataclass
from jax.tree_util import register_pytree_node_class
from pydantic import BaseModel

from .jax import jax, jnp
import numpy as np

from climate_health.external.models.jax_models.model_spec import Poisson, Exponential, Normal, LogNormal
from climate_health.external.models.jax_models.protoype_annotated_spec import Positive, Probability

state_or_param = lambda f: register_pytree_node_class(dataclasses.dataclass(f, frozen=True))

def get_normal_prior(field):
    if field.type == float:
        mu = 0
        if field.default is not None:
            mu = field.default
        return Normal(mu, 10)
    if field.type == Positive:
        mu = 1
        if field.default is not None:
            mu = field.default
        return LogNormal(np.log(mu), 10)


class PydanticTree:

    def tree_flatten(self):
        obj = self
        ret = tuple(getattr(obj, field.name) for field in dataclasses.fields(obj))
        #ret = ({field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}, None)
        return ret, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def sample(self, key, shape=()):
        obj = self
        d = {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}

        return self.__class__(**{name: obj.sample(key, shape=shape) if hasattr(obj, 'sample') else obj for key, (name, obj) in zip(jax.random.split(key, len(d)), d.items())})

    def __log_prob(self, value: 'PydanticTree'):
        return sum(getattr(self, field.name).log_prob(getattr(value, field.name))
                   for field in dataclasses.fields(self)
                   if hasattr(getattr(self, field.name), 'log_prob'))


def get_state_transform(params):
    new_fields = []

    converters = []
    identity = lambda x: x
    for field in dataclasses.fields(params):
        if field.type == Positive:
            converters.append(jnp.exp)
            default = Normal(np.log(field.default), 1.)
        elif issubclass(field.type, PydanticTree):
            T, f = get_state_transform(field.type)
            converters.append(f)
            default = T()
        else:
            converters.append(identity)
            default = Normal(field.default, 10.)
        new_fields.append((field.name, float, default))
    new_class = dataclasses.make_dataclass('T_' + params.__name__, new_fields, bases=(PydanticTree,), frozen=True)
    register_pytree_node_class(new_class)
    def f(transformed: new_class) -> params:
        return params.tree_unflatten(None, tuple(converter(val) for converter, val in zip(converters, transformed.tree_flatten()[0])))
    return new_class, f

@state_or_param
class SIRParams(PydanticTree):
    beta: Positive = 0.1#LogNormal(np.log(0.1), 1.)
    gamma: Positive = 0.05# LogNormal(np.log(0.05), 1.)


@state_or_param
class Params(PydanticTree):
    sir: SIRParams = SIRParams()
    observation_rate: Positive = 0.1


probabilities = dataclasses.dataclass


@state_or_param
class SIRState(PydanticTree):
    S: Probability
    I: Probability
    R: Probability


StateType = TypeVar('StateType')


@dataclasses.dataclass
class MarkovChain(Generic[StateType]):
    transition: Callable[[StateType], StateType]
    initial_state: SIRState
    time_period: np.ndarray

    def sample(self, shape: tuple[int]) -> StateType:
        def t(state, _):
            r = self.transition(state)
            return r, r

        states = jax.lax.scan(t, self.initial_state, None, length=len(self.time_period))
        return states[1]


def is_random(value):
    return hasattr(value, 'sample') and hasattr(value, 'log_prob')


def get_markov_chain(transition, initial_state, time_period):
    mc = MarkovChain(transition, initial_state, time_period)
    first = transition(initial_state)
    if not is_random(first):
        return mc.sample(())
    return mc


@bnpdataclass.bnpdataclass
class SIRObserved:
    population: int
    cases: int


next_state = lambda state, params: SIRState(
    S=state.S - state.S * params.beta * state.I,
    I=state.I + state.S * params.beta * state.I - state.I * params.gamma,
    R=state.R + state.I * params.gamma)


def main_sir(params: Params, observations: SIRObserved, key=jax.random.PRNGKey(0)):
    time_period = np.arange(len(observations))
    t = partial(next_state, params=params.sir)
    states = get_markov_chain(t, SIRState(S=0.9, I=0.1, R=0.0), time_period).sample(key)
    return Poisson(states.I * observations.population * params.observation_rate)


if __name__ == '__main__':
    params = SIRParams(0.1, 0.1)
    population = 1000
    cases = 100
    main_sir(params, SIRObserved(population, cases))

# observed[t] = Poisson(state[t].I * population * params.observation_rate)
# infected = state[t].I

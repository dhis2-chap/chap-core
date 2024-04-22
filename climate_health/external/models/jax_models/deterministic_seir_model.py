import dataclasses
from functools import partial
from typing import TypeVar, Generic
from bionumpy import bnpdataclass
from jax.tree_util import register_pytree_node_class
from pydantic import BaseModel

from .jax import jax
import numpy as np

from climate_health.external.models.jax_models.model_spec import Poisson, Exponential
from climate_health.external.models.jax_models.protoype_annotated_spec import Positive, Probability

state_or_param = lambda f: register_pytree_node_class(dataclasses.dataclass(f, frozen=True))


class PydanticTree:

    def tree_flatten(self):
        obj = self
        ret = tuple(getattr(obj, field.name) for field in dataclasses.fields(obj))
        #ret = ({field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}, None)
        return ret, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def sample(self, key):
        obj = self
        d = {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}

        return self.__class__(**{name: obj.sample(key) if hasattr(obj, 'sample') else obj for key, (name, obj) in zip(jax.random.split(key, len(d)), d.items())})


@state_or_param
class SIRParams(PydanticTree):
    beta: Positive = Exponential(0.1)
    gamma: Positive = Exponential(0.05)


@state_or_param
class Params(PydanticTree):
    sir: SIRParams = SIRParams()
    observation_rate: Positive = Exponential(0.1)


probabilities = dataclasses.dataclass


@state_or_param
class SIRState(PydanticTree):
    S: Probability
    I: Probability
    R: Probability


StateType = TypeVar('StateType')


@dataclasses.dataclass
class MarkovChain(Generic[StateType]):
    transition: callable
    initial_state: SIRState
    time_period: np.ndarray

    def sample(self, shape: tuple[int]) -> StateType:
        def t(state, _):
            r = self.transition(SIRState.tree_unflatten(None, state)).tree_flatten()[0]
            return r, r

        print(t(self.initial_state.tree_flatten()[0], None))
        states = jax.lax.scan(t, self.initial_state.tree_flatten()[0], None, length=len(self.time_period))
        return SIRState.tree_unflatten(None, states[1])


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


def main_sir(params: Params, observations: SIRObserved):
    time_period = np.arange(len(observations))
    t = partial(next_state, params=params.sir)
    states = get_markov_chain(t, SIRState(S=0.9, I=0.1, R=0.0), time_period)
    return Poisson(states.I * observations.population * params.observation_rate)


if __name__ == '__main__':
    params = SIRParams(0.1, 0.1)
    population = 1000
    cases = 100
    main_sir(params, SIRObserved(population, cases))

# observed[t] = Poisson(state[t].I * population * params.observation_rate)
# infected = state[t].I

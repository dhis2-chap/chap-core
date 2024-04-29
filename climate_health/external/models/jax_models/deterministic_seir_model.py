import operator as op
import dataclasses
from functools import partial
from typing import TypeVar, Generic, Callable
from bionumpy import bnpdataclass
from jax.tree_util import register_pytree_node_class

from climate_health.external.models.jax_models.utii import state_or_param, PydanticTree
from .jax import jax, jnp
import numpy as np

from climate_health.external.models.jax_models.model_spec import Poisson, Normal, distributionclass
from climate_health.external.models.jax_models.protoype_annotated_spec import Positive, Probability


@state_or_param
class SIRParams(PydanticTree):
    beta: Positive = 0.1  # LogNormal(np.log(0.1), 1.)
    gamma: Positive = 0.05  # LogNormal(np.log(0.05), 1.)

@state_or_param
class ProbSIRParams(SIRParams):
    diff_scale: Positive = 0.5

@state_or_param
class Params(PydanticTree):
    sir: SIRParams = SIRParams()
    observation_rate: Positive = 0.1


@state_or_param
class ProbabilisticParams(Params):
    sir: ProbSIRParams = ProbSIRParams()

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

    def sample(self, key, shape: tuple[int]=()) -> StateType:
        def t(state, _):
            r = self.transition(state)
            return r, r

        def random_t(state, t_key):
            r = self.transition(state).sample(t_key)
            return r, r
        if is_random(self.transition(self.initial_state)):
            states = jax.lax.scan(random_t, self.initial_state, jax.random.split(key, len(self.time_period)))
        else:
            states = jax.lax.scan(t, self.initial_state, None, length=len(self.time_period))
        return states[1]

    def log_prob(self, states):
        prev_state = states[:-1]
        next_state = states[1:]
        return self.transition(prev_state).log_prob(next_state).sum() + self.initial_state.log_prob(states[0])


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


T = TypeVar('T')


def apply_binary(f, a: PydanticTree, b: PydanticTree):
    return a.tree_unflatten(None, tuple(f(x, y) for x, y in zip(a.tree_flatten()[0], b.tree_flatten()[0])))


def transformed_diff_distribution(previous_state: T, expected_next_state: T, scale, t, inv_t):
    sub = lambda a, b: a.__class__(
        *(getattr(a, field.name) - getattr(b, field.name) for field in dataclasses.fields(a)))
    transformed = t(previous_state)
    expected_diffs = sub(t(expected_next_state), transformed)

    @distributionclass
    class Dist:
        def log_prob(self, new_state: T):
            d = apply_binary(op.sub, t(new_state), transformed)
            return sum(
                Normal(ed, scale).log_prob(od) for ed, od in zip(expected_diffs.tree_flatten()[0], d.tree_flatten()[0]))

        def sample(self, key, shape=()) -> T:
            t_sample = apply_binary(lambda x, y: x + Normal(y, scale).sample(key, shape), transformed, expected_diffs)
            return inv_t(t_sample)

    return Dist()


next_state = lambda state, params: SIRState(
    S=state.S - state.S * params.beta * state.I,
    I=state.I + state.S * params.beta * state.I - state.I * params.gamma,
    R=state.R + state.I * params.gamma)


def next_state_dist(state, params, t, inv_t):
    return transformed_diff_distribution(state, next_state(state, params), params.diff_scale, t, inv_t)


def main_sir(params: Params, observations: SIRObserved, key=jax.random.PRNGKey(0), transition_function=next_state):
    time_period = np.arange(len(observations))
    t = partial(transition_function, params=params.sir)
    states = get_markov_chain(t, SIRState(S=0.9, I=0.19, R=0.01), time_period).sample(key)
    return Poisson(states.I * observations.population * params.observation_rate), states


def get_categorical_transform(cls: PydanticTree) -> tuple[
    type, Callable[['cls'], PydanticTree], Callable[[PydanticTree], 'cls']]:
    new_fields = [(field.name, float)
                  for field in dataclasses.fields(cls)]
    new_class = dataclasses.make_dataclass('T_' + cls.__name__, new_fields, bases=(PydanticTree,), frozen=True)
    register_pytree_node_class(new_class)
    def f(x: cls) -> new_class:
        values = x.tree_flatten()[0]
        return new_class.tree_unflatten(None, [jnp.log(value) for value in values])

    def inv_f(x: new_class) -> cls:
        values = x.tree_flatten()[0]
        new_values = [jnp.exp(value) for value in values]
        s = sum(new_values)
        return cls.tree_unflatten(None, [value / s for value in new_values])

    return new_class, f, inv_f


if __name__ == '__main__':
    params = SIRParams(0.1, 0.1)
    population = 1000
    cases = 100

    main_sir(params, SIRObserved(population, cases))

# observed[t] = Poisson(state[t].I * population * params.observation_rate)
# infected = state[t].I

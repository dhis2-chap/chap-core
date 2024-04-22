from functools import partial

import numpy as np
import pytest
from jax.tree_util import tree_structure
import jax
import jax.random

from climate_health.external.models.jax_models.deterministic_seir_model import SIRParams, Params, SIRState, MarkovChain, \
    get_markov_chain, main_sir, SIRObserved, get_state_transform
from climate_health.external.models.jax_models.hmc import sample
import plotly.express as px
from jax.random import PRNGKey

import dataclasses

from climate_health.external.models.jax_models.model_spec import Exponential, LogNormal


def tree_log_prob(dist, value):
    if hasattr(dist, 'log_prob'):
        r = dist.log_prob(value)
        return r
    if hasattr(dist, '__dataclass_fields__'):
        return sum(tree_log_prob(getattr(dist, field.name), getattr(value, field.name))
                   for field in dataclasses.fields(dist))
    return 0

@pytest.mark.parametrize('a', [LogNormal(np.log(0.1), 3)])
def test_dist(a):
    samples = a.sample(PRNGKey(0), shape=(1000,))
    px.histogram(samples[samples<10], nbins=100).show()
    x = np.linspace(0, min(samples.max(), 10), 100)
    px.line(x=x, y=np.exp(a.log_prob(x))).show()

def test_state_transform():
    T, f = get_state_transform(SIRParams)
    print(dataclasses.fields(T))
    print(T)
    t = T()
    print(t)
    s = t.sample(PRNGKey(0), shape=())
    normal = f(s)
    print(normal)


def test_main_sir():
    T = 100
    params = Params(sir=SIRParams(beta=0.1, gamma=0.05), observation_rate=0.4)
    observations = SIRObserved(population=[100_000] * T, cases=[100] * T)
    rv = main_sir(params, observations)
    simulated = rv.sample(PRNGKey(0))
    prior = Params()
    f = partial(tree_log_prob, prior)
    px.line(simulated).show()

    def logprob_func(params):
        return main_sir(params, observations).log_prob(simulated).sum() + f(value=params)

    init_params = Params().sample(jax.random.PRNGKey(0))
    print(f(init_params))
    print(jax.value_and_grad(logprob_func)(init_params))

    samples = sample(logprob_func, PRNGKey(0), init_params, num_samples=100, num_warmup=200)
    px.line(samples.sir.beta).show()
    px.line(samples.sir.gamma).show()

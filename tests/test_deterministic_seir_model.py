from functools import partial

import numpy as np
import pytest
from jax.tree_util import tree_structure
import jax
import jax.random

from climate_health.external.models.jax_models.deterministic_seir_model import SIRParams, Params, SIRState, MarkovChain, \
    get_markov_chain, main_sir, SIRObserved, get_state_transform, transformed_diff_distribution, get_categorical_transform
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
    px.histogram(samples[samples < 10], nbins=100).show()
    x = np.linspace(0, min(samples.max(), 10), 100)
    px.line(x=x, y=np.exp(a.log_prob(x))).show()


def test_state_transform():
    T, f, inv_f = get_state_transform(SIRParams)
    t = T()
    s = t.sample(PRNGKey(0), shape=())
    normal = f(s)
    print(normal)

def test_state_transform_hierarchical():
    T, f,inv_f = get_state_transform(Params)
    t = T()
    s = t.sample(PRNGKey(0), shape=())
    normal = f(s)
    assert normal.observation_rate > 0
    assert normal.sir.beta > 0
    assert normal.sir.gamma > 0



def test_main_sir():
    T = 100
    params = Params(sir=SIRParams(beta=0.2, gamma=0.1), observation_rate=0.4)
    observations = SIRObserved(population=[100_000] * T, cases=[100] * T)
    rv = main_sir(params, observations)
    simulated = rv.sample(PRNGKey(0))
    T_Param, transform, inv_transform = get_state_transform(Params)
    T_prior = T_Param()
    # prior = Params()
    px.line(simulated).show()

    def logprob_func(raw_params):
        prior_prob = tree_log_prob(T_prior, raw_params)
        return main_sir(transform(raw_params), observations).log_prob(simulated).sum() + prior_prob#  prior_log_prob(value=params)

    init_params = T_Param().sample(jax.random.PRNGKey(0))
    print(jax.value_and_grad(logprob_func)(init_params))

    raw_samples = sample(logprob_func, PRNGKey(0), init_params, num_samples=100, num_warmup=100)
    samples = transform(raw_samples)
    px.line(samples.sir.beta).show()
    px.line(samples.sir.gamma).show()


def test_transformed_diff_distribution():
    state_1 = SIRState(S=0.9, I=0.19, R=0.01)
    state_2 = SIRState(S=0.8, I=0.19, R=0.01)
    T, f, inv_f = get_categorical_transform(SIRState)
    dist = transformed_diff_distribution(state_1, state_2, 1.0, f, inv_f)
    s = dist.sample(PRNGKey(0), shape=())
    print(s)
    print(dist.log_prob(s))
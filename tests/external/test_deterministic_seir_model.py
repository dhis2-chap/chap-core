import dataclasses
from functools import partial

import numpy as np
import plotly.express as px
import pytest

from climate_health.external.models.jax_models.deterministic_seir_model import SIRParams, Params, SIRState, main_sir, \
    SIRObserved, get_state_transform, transformed_diff_distribution, \
    get_categorical_transform, next_state_dist, ProbabilisticParams, ProbSIRParams
# from climate_health.external.models.jax_models.coin_svgd import CoinSVGD
from climate_health.external.models.jax_models.hmc import sample
from climate_health.external.models.jax_models.model_spec import LogNormal
import plotly
plotly.graph_objs.Figure.show = lambda self, *args, **kwargs: None

def tree_log_prob(dist, value):
    if hasattr(dist, 'log_prob'):
        r = dist.log_prob(value)
        return r
    if hasattr(dist, '__dataclass_fields__'):
        return sum(tree_log_prob(getattr(dist, field.name), getattr(value, field.name))
                   for field in dataclasses.fields(dist))
    return 0


@pytest.mark.parametrize('a', [LogNormal(np.log(0.1), 3)])
def test_dist(a, random_key):
    samples = a.sample(random_key, shape=(1000,))
    px.histogram(samples[samples < 10], nbins=100).show()
    x = np.linspace(0, min(samples.max(), 10), 100)
    px.line(x=x, y=np.exp(a.log_prob(x))).show()


def test_state_transform(random_key):
    T, f, inv_f = get_state_transform(SIRParams)
    t = T()
    s = t.sample(random_key, shape=())
    normal = f(s)
    print(normal)


def test_state_transform_hierarchical(random_key):
    T, f, inv_f = get_state_transform(Params)
    t = T()
    s = t.sample(random_key, shape=())
    normal = f(s)
    assert normal.observation_rate > 0
    assert normal.sir.beta > 0
    assert normal.sir.gamma > 0


def test_main_sir(random_key):
    T = 100
    params = Params(sir=SIRParams(beta=0.2, gamma=0.1), observation_rate=0.4)
    observations = SIRObserved(population=[100_000] * T, cases=[100] * T)

    rv, states = main_sir(params, observations)
    simulated = rv.sample(random_key)

    T_Param, transform, inv_transform = get_state_transform(Params)
    T_prior = T_Param()

    def logprob_func(raw_params):
        prior_prob = tree_log_prob(T_prior, raw_params)
        return main_sir(transform(raw_params), observations)[0].log_prob(
            simulated).sum() + prior_prob  # prior_log_prob(value=params)

    init_params = T_Param().sample(random_key)
    raw_samples = sample(logprob_func, random_key, init_params, num_samples=100, num_warmup=100)
    samples = transform(raw_samples)
    # px.line(samples.sir.beta).show()
    # px.line(samples.sir.gamma).show()


def coin_svgd_sample(log_prob_func, init_params, num_samples, n_iter):
    return CoinSVGD().coin_update(init_params, jax.grad(log_prob_func), n_iter)[n_iter]
    samples = svgd.run(n_iter)
    return samples


def test_probabilistic_sir(jax, random_key):
    T = 100
    params = ProbabilisticParams(sir=ProbSIRParams(beta=0.2, gamma=0.1, diff_scale=0.9), observation_rate=0.4)
    observations = SIRObserved(population=[100_000] * T, cases=[100] * T)
    T, f, inv_f = get_categorical_transform(SIRState)
    transition_function = partial(next_state_dist, t=f, inv_t=inv_f)
    rv, states = main_sir(params, observations, transition_function=transition_function)
    simulated = rv.sample(random_key)
    T_Param, transform, inv_transform = get_state_transform(ProbabilisticParams)
    T_prior = T_Param()

    def logprob_func(all_params):
        raw_params, raw_states = all_params
        prior_prob = tree_log_prob(T_prior, raw_params)
        states = inv_f(raw_states)
        state_prob = tree_log_prob(states, states)
        params = transform(raw_params)
        O, States = main_sir(params, observations)
        observation_prob = dataclasses.replace(O, rate=states.I*params.observation_rate*observations.population)
        return observation_prob.log_prob(simulated).sum() + state_prob + prior_prob

    init_params = T_Param().sample(random_key)
    init_states = main_sir(transform(init_params), observations)[1].sample(random_key)
    params = (init_params, f(init_states))
    raw_samples, _ = sample(logprob_func, random_key, params, num_samples=100, num_warmup=100)
    samples = transform(raw_samples)
    #trace_plot(samples.sir)
    #px.line(samples.sir.beta).show()
    #px.line(samples.sir.gamma).show()
    #px.line(samples.sir.diff_scale).show()
    #px.line(samples.observation_rate).show()


def trace_plot(samples):
    for field in dataclasses.fields(samples):
        obj = getattr(samples, field.name)
        px.line(obj, title=field.name).show()


def test_transformed_diff_distribution(random_key):
    state_1 = SIRState(S=0.9, I=0.19, R=0.01)
    state_2 = SIRState(S=0.8, I=0.19, R=0.01)
    T, f, inv_f = get_categorical_transform(SIRState)
    dist = transformed_diff_distribution(state_1, state_2, 1.0, f, inv_f)
    s = dist.sample(random_key, shape=())
    print(s)
    print(dist.log_prob(s))


def test_next_state_dist(random_key):
    params = ProbSIRParams(0.1, 0.1)
    state = SIRState(S=0.9, I=0.19, R=0.01)
    T, f, inv_f = get_categorical_transform(SIRState)
    dist = next_state_dist(state, params, f, inv_f)
    s = dist.sample(random_key, shape=())

    print(s)
    print(dist.log_prob(s))

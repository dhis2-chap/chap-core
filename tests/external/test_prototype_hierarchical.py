import plotly.express as px
import numpy as np
import pytest
from bionumpy.bnpdataclass import bnpdataclass

from climate_health.external.models.jax_models.hmc import sample
from climate_health.external.models.jax_models.prototype_hierarchical import linear_regression, Observations, \
    GlobalParams, get_logprob_func, DistrictParams, hierarchical_linear_regression, get_hierarchy_logprob_func, \
    GlobalSeasonalParams, SeasonalObservations, seasonal_linear_regression
from climate_health.external.models.jax_models.utii import get_state_transform, index_tree, tree_sample
from tests.external.test_deterministic_seir_model import trace_plot


def test_linear_regression(random_key, jax):
    x = np.arange(25, dtype=float)
    T_Param, transform, inv_transform = get_state_transform(GlobalParams)
    real_params = GlobalParams(2., 3., 5.)
    observed = Observations(x=x, y=x * 3 + 2)
    sampled_y = linear_regression(real_params, observed).sample(random_key)
    observed = Observations(observed.x, y=sampled_y)

    init_params= get_state_transform(GlobalParams)[0]().sample(random_key)
    logprob_func = get_logprob_func(GlobalParams, observed)
    raw_samples = sample(
        logprob_func, random_key, init_params, num_samples=100, num_warmup=100)
    trace_plot(transform(raw_samples), real_params)

@pytest.fixture
def seasonal_params():
    month_effect = np.zeros(12, dtype=float)
    month_effect[[6, 8]] = [10., -10.]
    params = GlobalSeasonalParams(2., 3., 1., month_effect)
    return params

@pytest.fixture
def seasonal_data(seasonal_params, random_key):
    T = 36
    x = np.arange(T) * 5.0 % 3
    month = np.arange(T) % 12
    model = seasonal_linear_regression(
        seasonal_params,
        SeasonalObservations(x=x, y=x*3.0, month=month))
    return SeasonalObservations(x=x, month=month, y=model.sample(random_key))


def test_seasonal_inference(seasonal_params, seasonal_data, random_key, jax):
    T_Param, transform, inv_transform = get_state_transform(GlobalSeasonalParams)
    real_params = seasonal_params
    observed = seasonal_data
    init_params = T_Param().sample(random_key)
    logprob_func = get_logprob_func(GlobalSeasonalParams, observed, seasonal_linear_regression)
    raw_samples = sample(logprob_func, random_key, init_params, num_samples=100, num_warmup=100)
    trace_plot(transform(raw_samples), real_params)



def test_sample_seasonal(random_key, jax, seasonal_params):
    T = 36
    x = np.arange(T) * 5 % 3
    month = np.arange(T) % 12
    model = seasonal_linear_regression(
        seasonal_params,
        SeasonalObservations(x=x, y=x*3, month=month))
    samples = model.sample(random_key)
    print(samples)


def test_seasonal_regression(random_key, jax):
    x = np.arange(25, dtype=float)
    T_Param, transform, inv_transform = get_state_transform(GlobalParams)
    real_params = GlobalParams(2., 3., 5.)
    observed = Observations(x=x, y=x * 3 + 2)
    sampled_y = linear_regression(real_params, observed).sample(random_key)
    observed = Observations(observed.x, y=sampled_y)

    init_params= get_state_transform(GlobalParams)[0]().sample(random_key)
    logprob_func = get_logprob_func(GlobalParams, observed)
    raw_samples = sample(
        logprob_func, random_key, init_params, num_samples=100, num_warmup=100)
    trace_plot(transform(raw_samples), real_params)

@pytest.fixture
def hierearchical_data():
    x = np.arange(10, dtype=float)
    return {location: Observations(x=x, y=x * float(b) + float(a))
            for location, a, b in zip(['A', 'B', 'C'],
                                   [2, 3, 4],
                                   [3, 4, 5])}



def test_hierarchical(hierearchical_data, random_key, jax):
    observed = hierearchical_data
    T_Param, transform, inv = get_state_transform(GlobalParams)
    T_ParamD, transformD, invD = get_state_transform(DistrictParams)
    real_params = GlobalParams(2., 3., 5.), {name: DistrictParams(float(i), float(i)) for i, name in enumerate(observed)}
    true_samples= tree_sample(hierarchical_linear_regression(real_params[0], real_params[1], observed), random_key)
    observed = {name: Observations(observed[name].x, y=true_samples[name]) for name in observed}
    logprob_func = get_hierarchy_logprob_func(GlobalParams, DistrictParams, observed)
    init_params = T_Param().sample(random_key), {location: T_ParamD().sample(random_key) for location in observed}
    #(jax.value_and_grad(logprob_func)(init_params))
    raw_samples = sample(logprob_func, random_key, init_params, num_samples=100, num_warmup=100)
    last_sample = index_tree(raw_samples, -1)
    print(logprob_func(last_sample))
    treal = (inv(real_params[0]), {name: invD(real_params[1][name]) for name in real_params[1]})
    print(treal)
    print(logprob_func(treal))
    trace_plot(transform(raw_samples[0]), real_params[0])
    for name, real in real_params[1].items():
        trace_plot(transformD(raw_samples[1][name]), real, name)

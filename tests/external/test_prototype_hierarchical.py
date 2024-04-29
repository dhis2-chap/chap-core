import plotly.express as px
import numpy as np
import pytest
from bionumpy.bnpdataclass import bnpdataclass

from climate_health.external.models.jax_models.hmc import sample
from climate_health.external.models.jax_models.prototype_hierarchical import linear_regression, Observations, \
    GlobalParams, get_logprob_func, DistrictParams
from climate_health.external.models.jax_models.utii import get_state_transform
from tests.external.test_deterministic_seir_model import trace_plot


def test_linear_regression(random_key, jax):
    x = np.arange(25, dtype=float)
    T_Param, transform, inv_transform = get_state_transform(GlobalParams)
    #prior = T_Param()
    real_params = GlobalParams(2., 3., 5.)
    observed = Observations(x=x, y=x * 3 + 2)
    sampled_y = linear_regression(real_params, observed).sample(random_key)
    observed = Observations(observed.x, y=sampled_y)
    # def logprob_func(t_params):
    #     all_params = transform(t_params)
    #     return prior.log_prob(t_params) + linear_regression(all_params, observed).log_prob(sampled_y).sum()
    #
    # init_params = T_Param().sample(random_key)

    # print(init_params, transform(init_params), prior, jax.value_and_grad(logprob_func)(init_params))
    init_params= get_state_transform(GlobalParams)[0]().sample(random_key)
    logprob_func = get_logprob_func(GlobalParams, observed)
    raw_samples = sample(logprob_func, random_key, init_params, num_samples=100, num_warmup=100)
    trace_plot(transform(raw_samples), real_params)

@pytest.fixture
def hierearchical_data():
    x = np.arange(10, dtype=float)
    return {location: Observations(x=x, y=x * b + a)
            for location, a, b in (['A', 'B', 'C'], [2, 3, 4], [3, 4, 5])}


def test_hierarchical(random_key, jax):
    observed = hierearchical_data()
    #x = np.arange(25, dtype=float)
    #observed = Observations(x=x, y=x * 3 + 2)
    T_Param, transform, inv_transform = get_state_transform(GlobalParams)
    prior = T_Param()
    real_params = GlobalParams(2., 3., 5.), {name: DistrictParams(i, i) for i, name in observed}

    def logprob_func(t_params):
        all_params = transform(t_params)
        return prior.log_prob(t_params) + linear_regression(all_params, observed).log_prob(sampled_y).sum()

    init_params = T_Param().sample(random_key)
    raw_samples = sample(logprob_func, random_key, init_params, num_samples=100, num_warmup=100)
    trace_plot(transform(raw_samples), real_params)

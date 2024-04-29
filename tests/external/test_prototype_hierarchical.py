import plotly.express as px
import numpy as np

from climate_health.external.models.jax_models.hmc import sample
from climate_health.external.models.jax_models.prototype_hierarchical import linear_regression, Observations, \
    GlobalParams
from climate_health.external.models.jax_models.utii import get_state_transform
from tests.external.test_deterministic_seir_model import trace_plot


def test_linear_regression(random_key, jax):
    x = np.arange(25, dtype=float)
    observed = Observations(x=x, y=x * 3 + 2)
    T_Param, transform, inv_transform = get_state_transform(GlobalParams)
    prior = T_Param()
    real_params = GlobalParams(2., 3., 5.)
    sampled_y = linear_regression(real_params, observed).sample(random_key)
    print(sampled_y)
    #px.scatter(x, sampled_y).show()
    def logprob_func(t_params):
        all_params = transform(t_params)
        return prior.log_prob(t_params) + linear_regression(all_params, observed).log_prob(sampled_y).sum()

    init_params = T_Param().sample(random_key)

    print(init_params, transform(init_params), prior,  jax.value_and_grad(logprob_func)(init_params))

    raw_samples = sample(logprob_func, random_key, init_params, num_samples=100, num_warmup=100)
    trace_plot(transform(raw_samples), real_params)


def test_hierarchical(random_key, jax):
    x = np.arange(25, dtype=float)
    observed = Observations(x=x, y=x * 3 + 2)
    T_Param, transform, inv_transform = get_state_transform(GlobalParams)
    prior = T_Param()
    real_params = GlobalParams(2., 3., 5.)
    sampled_y = linear_regression(real_params, observed).sample(random_key)
    print(sampled_y)
    #px.scatter(x, sampled_y).show()
    def logprob_func(t_params):
        all_params = transform(t_params)
        return prior.log_prob(t_params) + linear_regression(all_params, observed).log_prob(sampled_y).sum()

    init_params = T_Param().sample(random_key)

    print(init_params, transform(init_params), prior,  jax.value_and_grad(logprob_func)(init_params))

    raw_samples = sample(logprob_func, random_key, init_params, num_samples=100, num_warmup=100)
    trace_plot(transform(raw_samples), real_params)

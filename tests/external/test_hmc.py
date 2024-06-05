import pytest
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
from jax import tree_leaves

from climate_health.external.models.jax_models.jax import jax
from climate_health.external.models.jax_models.hmc import sample, multichain_sample
from climate_health.external.models.jax_models.utii import state_or_param, PydanticTree
from climate_health.training_control import TrainingControl, PrintingTrainingControl


@state_or_param
class NestedParams(PydanticTree):
    c: float = 0.0
    d: float = 0.0


@state_or_param
class DummyParam(PydanticTree):
    a: float = 0.0
    b: float = 0.0
    e: NestedParams = NestedParams()


@pytest.fixture
def init_params():
    return DummyParam(a=1.0, b=2.0, e=NestedParams(c=3.0, d=4.0))


def dummy_logprob_func(params):
    return params.a ** 2 - (params.b - 2) ** 3 + (params.e.c * params.e.d - 20) ** 2


@pytest.mark.parametrize("n_samples", [50, 77])
def test_sample(init_params, random_key, n_samples):
    training_control = PrintingTrainingControl()
    samples = sample(dummy_logprob_func, random_key, init_params, num_samples=n_samples, num_warmup=20,
                     training_control=training_control)
    assert training_control.get_progress() == 1.0
    assert samples.a.shape == (n_samples,)
    assert samples.b.shape == (n_samples,)
    assert samples.e.c.shape == (n_samples,)


def init_param_func(key):
    ratio = jax.random.uniform(key)
    return DummyParam(a=1.0 * ratio,
                      b=2.0 * ratio,
                      e=NestedParams(c=3.0 * ratio, d=4.0 * ratio))


def arviz_trace_from_states(states, info, burn_in=0):
    import arviz as az
    position = states.position
    if isinstance(position, jax.Array):  # if states.position is array of samples
        position = dict(samples=position)
    else:
        try:
            position = position._asdict()
        except AttributeError:
            pass

    samples = {}
    for param in position.keys():
        ndims = len(position[param].shape)
        if ndims >= 2:
            samples[param] = jnp.swapaxes(position[param], 0, 1)[
                :, burn_in:
            ]  # swap n_samples and n_chains
            divergence = jnp.swapaxes(info.is_divergent[burn_in:], 0, 1)

        if ndims == 1:
            divergence = info.is_divergent
            samples[param] = position[param]

    trace_posterior = az.convert_to_inference_data(samples)
    trace_sample_stats = az.convert_to_inference_data(
        {"diverging": divergence}, group="sample_stats"
        )
    trace = az.concat(trace_posterior, trace_sample_stats)
    return trace

def test_multichain_sample(random_key):
    samples = multichain_sample(dummy_logprob_func, random_key, init_param_func, num_samples=1000, num_warmup=1000,
                                n_chains=4)
    assert samples.a.shape == (1000, 4)
    print('ESS', [effective_sample_size(leaf_node) for leaf_node in tree_leaves(samples)])
    print('Rhat', [potential_scale_reduction(leaf_node) for leaf_node in tree_leaves(samples)])


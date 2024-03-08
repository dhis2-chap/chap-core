import jax
from jax import numpy as jnp
from probabilistic_machine_learning.cases.diff_model import MosquitoModelSpec
from probabilistic_machine_learning.cases.hybrid_model import HybridModel
from probabilistic_machine_learning.cases.multilevel_model import MultiLevelModelSpecFactory

from .state_space_model import SimpleSampler


def main_model():
    model_spec = MosquitoModelSpec
    model_spec = MultiLevelModelSpecFactory.from_period_lengths(model_spec, periods_lengths)
    model = HybridModel(model_spec)
    sampler = SimpleSampler.from_model(model,
                                       jax.random.PRNGKey(40), n_warmup_samples=n_warmup_samples,
                                       n_samples=n_samples)
    transformed_states = jnp.array([model_spec.state_transform(model_spec.init_state)] * (T // 100))
    init_diffs = model.sample_diffs(
        transition_key=jax.random.PRNGKey(10000), params=model_spec.good_params,
        exogenous=climate_data.max_temperature)
    sampler.train(data_set,
                  init_values=model_spec.good_params | {
                      'logits_array': init_diffs,
                      'transformed_states': transformed_states})
from jax import tree_flatten, tree_leaves, tree_unflatten
import jax.numpy as jnp
from .jax import jax, blackjax
from dynamax.utils.utils import pytree_stack
import logging

logger = logging.getLogger(__name__)


class TrainingControl:
    def __init__(self, num_samples, num_warmup):
        self._num_samples = num_samples
        self._num_warmup = num_warmup

    def register_progress(self, n_sampled):
        pass


def pytree_concatenate(pytrees):
    _, treedef = tree_flatten(pytrees[0])
    leaves = [tree_leaves(tree) for tree in pytrees]
    return tree_unflatten(treedef, [jnp.concatenate(vals) for vals in zip(*leaves)])


def inference_loop(rng_key, kernel, initial_state, num_samples):
    epoch_size = num_samples // 10

    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    all_states = []
    last_key, *base_keys = jax.random.split(rng_key, num_samples // epoch_size + 1)
    for base_key in base_keys:
        keys = jax.random.split(base_key, epoch_size)
        initial_state, states = jax.lax.scan(one_step, initial_state, keys)
        all_states.append(states)
    if num_samples % epoch_size != 0:
        keys = jax.random.split(last_key, num_samples % epoch_size)
        _, states = jax.lax.scan(one_step, initial_state, keys)
        all_states.append(states)
    return pytree_concatenate(all_states)


def sample(logdensity, rng_key, initial_position, num_samples=1000, num_warmup=1000, training_control=None):
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    logger.info("Starting warmup")
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=num_warmup)
    logger.info("Warmup done")
    kernel = blackjax.nuts(logdensity, **parameters).step
    states = inference_loop(sample_key, kernel, state, num_samples)
    mcmc_samples = states.position
    return mcmc_samples

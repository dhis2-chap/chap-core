from typing import Any, Optional

from jax import tree_flatten, tree_leaves, tree_unflatten
import jax.numpy as jnp

from climate_health.training_control import TrainingControl
from .jax import jax, blackjax
import logging

logger = logging.getLogger(__name__)


def pytree_concatenate(pytrees):
    _, treedef = tree_flatten(pytrees[0])
    leaves = [tree_leaves(tree) for tree in pytrees]
    return tree_unflatten(treedef, [jnp.concatenate(vals) for vals in zip(*leaves)])


def inference_loop(rng_key, kernel, initial_state: Any, num_samples: int, training_control: Optional[
    TrainingControl] = None):
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
        if training_control is not None:
            training_control.register_progress(epoch_size)
            if training_control.is_cancelled():
                return
    if num_samples % epoch_size != 0:
        keys = jax.random.split(last_key, num_samples % epoch_size)
        _, states = jax.lax.scan(one_step, initial_state, keys)
        all_states.append(states)
        if training_control is not None:
            training_control.register_progress(num_samples % epoch_size)
    return pytree_concatenate(all_states)


def sample(logdensity, rng_key, initial_position, num_samples=1000, num_warmup=1000, training_control=None):
    if training_control is None:
        training_control = TrainingControl()
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    logger.info("Starting warmup")
    training_control.set_total_samples(num_samples+num_warmup)
    training_control.set_status("Warming up")
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=num_warmup)
    training_control.register_progress(num_warmup)
    logger.info("Warmup done")
    kernel = blackjax.nuts(logdensity, **parameters).step
    training_control.set_status("Sampling")
    states = inference_loop(sample_key, kernel, state, num_samples, training_control=training_control)
    mcmc_samples = states.position
    return mcmc_samples

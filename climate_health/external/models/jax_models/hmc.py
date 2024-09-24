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


def inference_loop(
    rng_key,
    kernel,
    initial_state: Any,
    num_samples: int,
    training_control: Optional[TrainingControl] = None,
):
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
    if num_samples % epoch_size != 0:
        keys = jax.random.split(last_key, num_samples % epoch_size)
        _, states = jax.lax.scan(one_step, initial_state, keys)
        all_states.append(states)
        if training_control is not None:
            training_control.register_progress(num_samples % epoch_size)
    return pytree_concatenate(all_states)


def inference_loop_multiple_chains(
    rng_key, initial_states, tuned_params, log_prob_fn, num_samples, num_chains
):
    kernel = blackjax.nuts.build_kernel()

    def step_fn(key, state, **params):
        return kernel(key, state, log_prob_fn, **params)

    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, infos = jax.vmap(step_fn)(keys, states, **tuned_params)
        return states, (states, infos)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

    return (states, infos)


def multichain_sample(
    logdensity, rng_key, init_param_func, num_samples=1000, num_warmup=1000, n_chains=4
):
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    # we use 4 chains for sampling
    rng_key, init_key, warmup_key = jax.random.split(rng_key, 3)
    init_keys = jax.random.split(init_key, n_chains)
    init_params = jax.vmap(init_param_func)(init_keys)

    @jax.vmap
    def call_warmup(seed, param):
        (initial_states, tuned_params), _ = warmup.run(
            seed, param, num_steps=num_warmup
        )
        return initial_states, tuned_params

    warmup_keys = jax.random.split(warmup_key, n_chains)

    initial_states, tuned_params = call_warmup(warmup_keys, init_params)
    n_samples = 1000
    rng_key, sample_key = jax.random.split(rng_key)
    states, infos = inference_loop_multiple_chains(
        sample_key, initial_states, tuned_params, logdensity, n_samples, n_chains
    )
    return states.position


def sample(
    logdensity,
    rng_key,
    initial_position,
    num_samples=1000,
    num_warmup=1000,
    training_control=None,
):
    if training_control is None:
        training_control = TrainingControl()
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    training_control.set_total_samples(num_samples + num_warmup)
    training_control.set_status("Warming up")
    (state, parameters), _ = warmup.run(
        warmup_key, initial_position, num_steps=num_warmup
    )
    training_control.register_progress(num_warmup)
    kernel = blackjax.nuts(logdensity, **parameters).step
    training_control.set_status("Sampling")
    states = inference_loop(
        sample_key, kernel, state, num_samples, training_control=training_control
    )
    mcmc_samples = states.position
    return mcmc_samples

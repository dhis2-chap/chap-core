from .jax import jax, blackjax
import logging
logger = logging.getLogger(__name__)


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def sample(logdensity, rng_key, initial_position, num_samples=1000, num_warmup=1000):
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    logger.info("Starting warmup")
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=num_warmup)
    logger.info("Warmup done")
    kernel = blackjax.nuts(logdensity, **parameters).step
    states = inference_loop(sample_key, kernel, state, num_samples)
    mcmc_samples = states.position
    return mcmc_samples

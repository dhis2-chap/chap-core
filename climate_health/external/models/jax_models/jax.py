
try:
    from jax.scipy import stats
    from jax.random import PRNGKey
    import jax.numpy as jnp
    import jax
    import blackjax
except ImportError as e:
    jax, jnp, stats, PRNGKey = (None, None, None, None)
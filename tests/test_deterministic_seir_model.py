from functools import partial
from jax.tree_util import tree_structure
import jax
import jax.random

from climate_health.external.models.jax_models.deterministic_seir_model import SIRParams, Params, SIRState, MarkovChain, \
    get_markov_chain, main_sir, SIRObserved
from climate_health.external.models.jax_models.hmc import sample
import plotly.express as px
from jax.random import PRNGKey


def test_main_sir():
    T = 10
    params = Params(sir=SIRParams(beta=0.1, gamma=0.05), observation_rate=0.4)
    observations = SIRObserved(population=[100_000] * T, cases=[100] * T)
    rv = main_sir(params, observations)
    simulated = rv.sample(PRNGKey(0))
    logprob_func = lambda params: main_sir(params, observations).log_prob(simulated).sum()
    logprob_func(Params().sample(jax.random.PRNGKey(0)))

    init_params = Params().sample(jax.random.PRNGKey(0))
    t, aux = init_params.tree_flatten()
    sample(logprob_func, PRNGKey(0), init_params, num_samples=10, num_warmup=10)

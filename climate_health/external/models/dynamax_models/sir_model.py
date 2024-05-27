from dynamax.nonlinear_gaussian_ssm import ParamsNLGSSM, extended_kalman_smoother
from jax import tree_unflatten
import jax.numpy as jnp


from jaxtyping import Float

from climate_health.external.models.jax_models.model_spec import Normal
from climate_health.external.models.jax_models.utii import PydanticTree
from jax.scipy.special import logsumexp


class SIRParams:
    beta: Float
    gamma: Float
    a: Float


class LogitStates(PydanticTree):
    S: Float
    I: Float
    R: Float


def dynamic_function(prev_state: LogitStates, params):  # , beta: Float, gamma: Float):
    delta_1 = prev_state.S + prev_state.I + params.beta
    delta_2 = prev_state.I + params.gamma
    delta_3 = prev_state.R + params.gamma
    return LogitStates(S=logsumexp([prev_state.S, delta_1, delta_3], b=[1, -1, 1]),
                       I=logsumexp([prev_state.I, delta_1, delta_2], b=[1, 1, -1]),
                       R=logsumexp([prev_state.R, delta_2, delta_3], b=[1, 1, -1]))


def arrify_function(func, state_class: type):
    def new_func(array_state, params):
        return jnp.array(tree_unflatten(func(state_class(*array_state), params))

    return new_func

f = arrify_function(dynamic_function, LogitStates)
init_state = jnp.array([1, 0, -1])
def log_prob_func(params):
    ekf_params = ParamsNLGSSM(
        initial_mean=init_state,
        initial_covariance=jnp.eye(3),
        dynamics_function=f,
        dynamics_covariance=jnp.eye(3),
        emission_function=lambda state: state[1]+params.observation_rate,
        emission_covariance=1.0)

    return extended_kalman_smoother(ekf_params, obs).marginal_loglik

def sample(T, params=SIRParams(beta=0.1, gamma=0.1, a=0.1)):
    t = lambda state, state: Normal(dynamic_function(state[1], params)).sample()


def test_simple():
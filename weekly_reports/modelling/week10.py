''' Goal for this week is to reparameterize the mosquito model to have diffs in state space as hidden states which will 
hopefully speed up the inference enough to be easy to work with. We are also starting an iteration on the teamproject
so hopefully functionality for running and evaluating the models on real data will be in place soon.
'''
from probabilistic_machine_learning.cases.diff_model import MosquitoModelSpec
import jax.numpy as jnp
from jax.scipy.special import expit, logit

'''
def _get_full_diffs(P, logits, states):
    human_state, mosquito_state = states[..., :4], states[..., 4:]
    human_logits = logits[..., :4]
    human_diffs = human_state * expit(human_logits)
    mosquito_death_rate = _get_mosquito_death_rate(mosquito_state, P)
    death_rates = mosquito_death_rate
    mosquito_state = tuple(s * (1 - expit(d)) for s, d in zip(mosquito_state.T, death_rates))
    mosquito_logits = logits[..., 4:]
    mosquito_diffs = tuple(ms * expit(lg) for ms, lg in zip(mosquito_state, mosquito_logits.T))
    new_eggs = jnp.exp(P['log_eggrate']) * sum(mosquito_state[3:]) * expit(mosquito_logits.T[-1])
    full_human_diffs = - human_diffs + jnp.roll(human_diffs, 1, axis=-1)
    new_state = human_state+full_human_diffs
    md0 = mosquito_diffs[0] + new_eggs
    md1 = mosquito_diffs[1] + mosquito_diffs[0]
    md2 = mosquito_diffs[2] + mosquito_diffs[1]
    md3 = mosquito_diffs[3] + mosquito_diffs[2]
    md4 = mosquito_diffs[4] + mosquito_diffs[3]
    md5 = mosquito_diffs[4]
    new_state = jnp.array(
        [new_state[..., 0], new_state[..., 1], new_state[..., 2], new_state[..., 3],
         mosquito_state[0] - md0
         mosquito_state[1] - md1,
         mosquito_state[2] - md2,
         mosquito_state[3] - md3,
         mosquito_state[4] - md4,
         mosquito_state[5] + md5]).T
    return new_state, new_state
'''

class StateSpaceDiffMosquitoModelSpec(MosquitoModelSpec):
    pass

def test_refactor_transition():
    '''Refactor the transition function to explicitly calculate the diffs and the new state.'''
    old_spec = MosquitoModelSpec(MosquitoModelSpec.good_params)
    new_spec = StateSpaceDiffMosquitoModelSpec(MosquitoModelSpec.good_params)
    states = jnp.array([old_spec.init_state]*5)
    logits = jnp.linspace(-1, 1, 50).reshape(-1, 10)
    old_result = old_spec.transition(states, logits)
    new_result = new_spec.transition(states, logits)
    assert jnp.allclose(old_result[0], new_result[0])



def test_reparameterize_to_state_space_diffs():
    '''Reparameterize the model to have diffs in state space as hidden states.
    This makes the reconstruction step a simple cumsum and probably makes the autodiff more stable.
    Also move modelling code into climate_health repo to avoid discrepancies
    '''
    def sample_diffs(states, exogenous, params):
        human_state, mosquito_state = states[..., :4], states[..., 4:]
        human_logits = logits[..., :4]
        human_diffs = human_state * expit(human_logits)
        mosquito_death_rate = _get_mosquito_death_rate(mosquito_state, P)
        death_rates = mosquito_death_rate
        mosquito_state = tuple(s * (1 - expit(d)) for s, d in zip(mosquito_state.T, death_rates))
        mosquito_logits = logits[..., 4:]
        mosquito_diffs = tuple(ms * expit(lg) for ms, lg in zip(mosquito_state, mosquito_logits.T))
        new_eggs = jnp.exp(P['log_eggrate']) * sum(mosquito_state[3:]) * expit(mosquito_logits.T[-1])
        new_state = human_state - human_diffs + jnp.roll(human_diffs, 1, axis=-1)
        new_state = jnp.array(
            [new_state[..., 0], new_state[..., 1], new_state[..., 2], new_state[..., 3],
             mosquito_state[0] - mosquito_diffs[0] + new_eggs,
             mosquito_state[1] - mosquito_diffs[1] + mosquito_diffs[0],
             mosquito_state[2] - mosquito_diffs[2] + mosquito_diffs[1],
             mosquito_state[3] - mosquito_diffs[3] + mosquito_diffs[2],
             mosquito_state[4] - mosquito_diffs[4] + mosquito_diffs[3],
             mosquito_state[5] + mosquito_diffs[4]]).T

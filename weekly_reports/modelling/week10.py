''' Goal for this week is to reparameterize the mosquito model to have diffs in state space as hidden states which will 
hopefully speed up the inference enough to be easy to work with. We are also starting an iteration on the teamproject
so hopefully functionality for running and evaluating the models on real data will be in place soon.
'''
import numpy as np
from probabilistic_machine_learning.cases.diff_model import MosquitoModelSpec, _get_mosquito_death_rate
import jax.numpy as jnp
from jax.scipy.special import expit, logit


def _new_transition(P, logits, states):
    full_human_diffs, full_mosquito_diffs, human_state, mosquito_state_after_deaths = _get_full_diffs(P, logits, states)
    new_mosquito_state = [a + b for a, b in zip(mosquito_state_after_deaths, full_mosquito_diffs)]
    new_human_state = human_state + full_human_diffs
    new_state = jnp.array([new_human_state[..., 0], new_human_state[..., 1], new_human_state[..., 2], new_human_state[..., 3]] + new_mosquito_state).T
    return new_state, new_state


def _get_full_diffs(P, logits, states):
    human_state, mosquito_state = states[..., :4], states[..., 4:]
    human_logits = logits[..., :4]
    human_diffs = human_state * expit(human_logits)
    full_human_diffs = -human_diffs + jnp.roll(human_diffs, 1, axis=-1)
    mosquito_death_rate = _get_mosquito_death_rate(mosquito_state, P)
    death_rates = mosquito_death_rate
    mosquito_deaths = tuple(s * expit(d) for s, d in zip(mosquito_state.T, death_rates))
    mosquito_state_after_deaths = tuple(s - d for s, d in zip(mosquito_state.T, mosquito_deaths))
    mosquito_logits = logits[..., 4:]
    mosquito_diffs = tuple(ms * expit(lg) for ms, lg in zip(mosquito_state_after_deaths, mosquito_logits.T))
    new_eggs = jnp.exp(P['log_eggrate']) * sum(mosquito_state_after_deaths[3:]) * expit(mosquito_logits.T[-1])
    full_mosquito_diffs = [new_eggs - mosquito_diffs[0]] + [mosquito_diffs[i - 1] - mosquito_diffs[i] for i in
                                                            range(1, 5)] + [mosquito_diffs[4]]
    full_mosquito_diffs = [d-death for d, death in zip(full_mosquito_diffs, mosquito_deaths)]
    return full_human_diffs, full_mosquito_diffs, human_state, mosquito_state.T


class StateSpaceDiffMosquitoModelSpec(MosquitoModelSpec):
    def transition(self, states, logits):
        P = self._params
        return _new_transition(P, logits, states)

    def get_state_diffs(self, diffs, states):
        '''Get the diffs in state space from the diffs in human readable space'''
        assert diffs.shape == states.shape, (diffs.shape, states.shape)
        new_state = jnp.array(diffs) + states
        old_state = self.state_transform(states)
        new_state = self.state_transform(new_state)
        return (new_state - old_state)


def test_refactor_transition():
    '''Refactor the transition function to explicitly calculate the diffs and the new state.'''
    T = 2
    old_spec = MosquitoModelSpec(MosquitoModelSpec.good_params)
    new_spec = StateSpaceDiffMosquitoModelSpec(MosquitoModelSpec.good_params)
    states = jnp.array([old_spec.init_state] * T)
    logits = jnp.linspace(-1, 1, T*10).reshape(-1, 10)
    old_result = old_spec.transition(states, logits)
    new_result = new_spec.transition(states, logits)
    np.testing.assert_allclose(old_result[0], new_result[0], rtol=1e-5)# , (new_result[0][0:2, :2], old_result[0][0:2, :2])

def test_state_space_diffs():
    '''Try to convert diffs in human space to diffs in state space (logit/log space)'''
    T = 2
    old_spec = MosquitoModelSpec(MosquitoModelSpec.good_params)
    new_spec = StateSpaceDiffMosquitoModelSpec(MosquitoModelSpec.good_params)
    states = jnp.array([old_spec.init_state] * T)
    logits = jnp.linspace(-1, 1, T*10).reshape(-1, 10)

    true_new_state = old_spec.transition(states, logits)[0]
    diffs = _get_full_diffs(new_spec._params, logits, states)
    state_diffs = new_spec.get_state_diffs(diffs, states)
    new_state = new_spec.state_transform(states+state_diffs)
    new_h_state = new_spec.inverse_state_transform(new_state)
    np.testing.assert_allclose(true_new_state, new_h_state, rtol=1e-5)

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

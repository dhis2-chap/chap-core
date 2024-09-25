'''
Should get the mosquito model incorporated into the human model.
 With that and different scale for health and climate data, it is ready to use on real data.

- Inlcude Mosquito compartments in the model
- Use data from multiple locations
- Use weather data on finer resolution than health data (i.e. daily weather data and monthly health data)
'''
import time

import numpy as np
from matplotlib import pyplot as plt
from probabilistic_machine_learning.cases.diff_model import MosquitoModelSpec, DiffModel
from probabilistic_machine_learning.cases.hybrid_model import HybridModel
from probabilistic_machine_learning.cases.multilevel_model import MultiLevelModelSpecFactory
from probabilistic_machine_learning.cases.diff_encoded_mosquito import full_model
from report_tests import show

import jax
import jax.numpy as jnp

from scipy.special import logit, expit

from chap_core.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData
from chap_core.plotting.prediction_plot import prediction_plot, forecast_plot
from tests import EXAMPLE_DATA_PATH
from weekly_reports.modelling.week8 import get_simulator
from chap_core.external.models.jax_models.state_space_model import SimpleSampler

ClimateHealthTimeSeries._assert_same_lens = lambda *_, **__: None
HealthData._assert_same_lens = lambda *_, **__: None

def forecast_plot(true_data: HealthData, predicition_sampler, climate_data: ClimateData, n_samples):
    samples = np.array([predicition_sampler.sample(climate_data) for _ in range(n_samples)])
    plt.plot(samples.T, color='grey', alpha=0.1)
    plt.plot(true_data.disease_cases, color='blue')
    return plt.gcf()
    #return plot_forecast(quantiles, true_data)

def test_human_mosquito_model():
    '''Find a good parameter space for human/mosquito model'''

    (sample, log_prob, reconstruct_state, sample_diffs), (param_names, n_states) = full_model()
    print(param_names)
    climate_data = ClimateData.from_csv(EXAMPLE_DATA_PATH / 'climate_data_daily.csv')[:365 * 5]
    simulator = get_simulator(sample, param_names)
    mosquito_data = simulator.simulate(climate_data)
    plt.plot(mosquito_data.disease_cases)
    return show(plt.gcf())


def test_investigate_infection_rate_function():
    '''
    Should be quite low for 0 infected humans and rise approximately linearly with the number of infected humans.
    '''
    alpha = logit(0.0001)
    x = np.linspace(0, 1, 100)
    for beta in [0.1, 0.5, 1, 2, 10]:
        plt.plot(x, expit(alpha + beta * np.log(x)), label=f'beta={beta}')
    plt.legend()
    plt.title('Infection rate as a function of infected humans')
    return show(plt.gcf())


def test_ivestigate_human_inection_rate_function():
    '''
    Should be quite low for 0 infected mosquitoes and rise approximately linearly with the number of infected mosquitoes.
    '''
    alpha = logit(0.0001)
    x = np.linspace(0, 1000000, 100)
    for beta in [0.1, 0.5, 1, 2, 10]:
        plt.plot(x, expit(alpha + beta * np.log(x)), label=f'beta={beta}')
    plt.legend()
    plt.title('Infection rate as a function of infected mosquitoes')
    return show(plt.gcf())


def test_human_mosquito_model_estimation():
    '''Try to estimate parameters for a well parameterized full model. This is not really estimating since we feed the true parameters to the model, but it is a good test of the capacity of the model to fit the data.
    '''
    (sample, log_prob, reconstruct_state, sample_diffs), (real_params, n_states) = full_model()
    for T in [100, 200]:
        fig = evaluate_human_mosq_model(log_prob, n_states, real_params, sample, sample_diffs, T)
    return show(fig)


def test_human_mosquito_model_estimation_more_Ts():
    '''Try to estimate parameters for a well parameterized full model. This is not really estimating since we feed the true parameters to the model, but it is a good test of the capacity of the model to fit the data.
    It seems that for large T are some numerical issues. The dependency on the first diff might be too big (too many steps), and gradient calculation becomes too unstable.
    Maybe we should break the dependencies by sampling actual states instead of diffs at fixed time intervals.
    '''
    (sample, log_prob, reconstruct_state, sample_diffs), (real_params, n_states) = full_model()
    for T in [10, 100, 500, 1000]:
        fig = evaluate_human_mosq_model(log_prob, n_states, real_params, sample, sample_diffs, T)
    return show(fig)


def evaluate_human_mosq_model(log_prob, n_states, real_params, sample, sample_diffs, T):
    climate_data = ClimateData.from_csv(EXAMPLE_DATA_PATH / 'climate_data_daily.csv')[:T]
    simulator = get_simulator(sample, real_params)
    health_data = simulator.simulate(climate_data)
    data_set = ClimateHealthTimeSeries.combine(health_data, climate_data)
    sampler = SimpleSampler(jax.random.PRNGKey(0),
                            log_prob, sample,
                            real_params, n_states,
                            n_warmup_samples=500)
    sampler.train(data_set,
                  init_values=real_params | {'init_diffs': sample_diffs(transition_key=jax.random.PRNGKey(10000),
                                                                        params=real_params,
                                                                        exogenous=climate_data.max_temperature)})
    fig = prediction_plot(health_data, sampler, climate_data, 10)
    return fig


class SimpleSimulator:
    def __init__(self, model):
        self.model = model

    def simulate(self, climate_data: ClimateData):
        T = len(climate_data) + 1
        samples = self.model.sampler(T, jax.random.PRNGKey(0), self.model.spec_class.good_params,
                                     climate_data.max_temperature)
        return HealthData(time_period=climate_data.time_period, disease_cases=samples)

    @classmethod
    def from_model(cls, model):
        return cls(model)


def test_refactor_mosquito_model():
    ''' Needs some refactoring before making the hybrid model. Time to use classes'''
    T = 100
    model_spec = MosquitoModelSpec
    model = DiffModel(model_spec)
    simulator = SimpleSimulator.from_model(model)
    climate_data = ClimateData.from_csv(EXAMPLE_DATA_PATH / 'climate_data_daily.csv')[:T]
    health_data = simulator.simulate(climate_data)
    plt.plot(health_data.disease_cases);
    plt.show()
    data_set = ClimateHealthTimeSeries.combine(health_data, climate_data)
    sampler = SimpleSampler.from_model(model, jax.random.PRNGKey(0))
    sampler.train(data_set,
                  init_values=model_spec.good_params | {
                      'init_diffs': model.sample_diffs(transition_key=jax.random.PRNGKey(10000),
                                                       params=model_spec.good_params,
                                                       exogenous=climate_data.max_temperature)})
    return show(prediction_plot(health_data, sampler, climate_data, 10))


def test_scan_with_fixed_values():
    '''Find out how to use scan over 2d array with init values'''
    init_values = jnp.array([0.9, 0.7, 0.2])
    diffs = jnp.array([[0.1, 0.3, 0.1], [0.2, 0.1, 0.1], [0.1, 0.1, 0.1]])

    def transition(state, diff):
        new_state = state + diff
        return new_state, new_state

    val = jax.lax.scan(transition, init_values, diffs)[1]
    print(val)


def test_scan_with_fixed_values_3d():
    '''Find out how to use scan over 2d array with init values, n_states=2, n_fixed=3, T=9'''
    init_values = jnp.array([[0.9, 0.1],
                             [0.7, 0.1],
                             [0.7, 0.2]])
    diffs = jnp.array([[[0.1, 0.2], [0.3, 0.4], [0.1, 0.3], [0.5, 0.5]],
                       [[0.2, 0.2], [0.1, 0.1], [0.1, 0.1], [0.5, 0.5]],
                       [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.5, 0.5]]])
    diffs = jnp.swapaxes(diffs, 0, 1)

    def transition(state, diff):
        new_state = state + diff
        return new_state, new_state

    val = jax.lax.scan(transition, init_values, diffs)[1]
    val = jnp.swapaxes(val, 0, 1).reshape(-1, init_values.shape[-1])
    print(val)


def test_hybrid_central_noncentral_model():
    '''
    This seems to maybe work, but requires alot of warmup samples to get the right parameters.
    Might need to speed up the sampling, but promising results.
    '''

    return check_hybrid_model_capacity(T=100, n_warmup_samples=200, n_samples=100)
    # (sample, log_prob, reconstruct_state, sample_diffs), (real_params, n_states) = simple_hybrid_model()


def check_hybrid_model_capacity(T=400, periods_lengths=None, n_warmup_samples=1000, n_samples=1000, model_spec = MosquitoModelSpec):
    if periods_lengths is not None:
        model_spec = MultiLevelModelSpecFactory.from_period_lengths(model_spec, periods_lengths)
    model = HybridModel(model_spec)
    simulator = SimpleSimulator.from_model(model)
    climate_data = ClimateData.from_csv(EXAMPLE_DATA_PATH / 'climate_data_daily.csv')[:T]
    health_data = simulator.simulate(climate_data)
    data_set = ClimateHealthTimeSeries.combine(health_data, climate_data)
    sampler = SimpleSampler.from_model(model, jax.random.PRNGKey(40), n_warmup_samples=n_warmup_samples,
                                       n_samples=n_samples)
    transformed_states = jnp.array([model_spec.state_transform(model_spec.init_state)] * (T // 100))
    init_diffs = model.sample_diffs(transition_key=jax.random.PRNGKey(10000), params=model_spec.good_params,
                               exogenous=climate_data.max_temperature)
    sampler.train(data_set,
                  init_values=model_spec.good_params | {
                      'logits_array': init_diffs,
                      'transformed_states': transformed_states})
    return show(forecast_plot(health_data, sampler, climate_data, 10))


def test_multilevel_model():
    '''Test functionality with monthly disease observations and daily weather data'''
    T = 100
    return check_hybrid_model_capacity(T=T, periods_lengths=jnp.full(T//20, 20), n_warmup_samples=100, n_samples=100)


def test_speedup_transitions():
    '''
    Check if we can get significant speedup by leveraging jax.jit:
    Did not manage to speed it up. Will try to reparameterize ot make state reconstruction faster.
    '''
    T = 400
    model_spec = MosquitoModelSpec
    model = HybridModel(model_spec)
    climate_data = ClimateData.from_csv(EXAMPLE_DATA_PATH / 'climate_data_daily.csv')[:T]
    diffs = model.sample_diffs(transition_key=jax.random.PRNGKey(10000),
                               params=model_spec.good_params,
                               exogenous=climate_data.max_temperature)
    transformed_states = jnp.array([model_spec.init_state] * (T // 100))
    t = time.time()
    model.recontstruct_state(diffs, transformed_states, params=model_spec.good_params)
    print(time.time() - t) #0.20 seconds
    # Did not manage to speed it up with jit

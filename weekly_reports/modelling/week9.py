'''
Should get the mosquito model incorporated into the human model.
 With that and different scale for health and climate data, it is ready to use on real data.

- Inlcude Mosquito compartments in the model
- Use data from multiple locations
- Use weather data on finer resolution than health data (i.e. daily weather data and monthly health data)
'''
import numpy as np
from matplotlib import pyplot as plt
from probabilistic_machine_learning.cases.diff_model import MosquitoModelSpec, DiffModel
from report_tests import show

import jax
from probabilistic_machine_learning.cases.diff_encoded_mosquito import pure_mosquito_model, full_model
# from probabilistic_machine_learning.cases.hybrid_model import simple_hybrid_model
from scipy.special import logit, expit

from climate_health.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData
from climate_health.plotting.prediction_plot import prediction_plot
from tests import EXAMPLE_DATA_PATH
from .week8 import check_model_capacity, get_parameterized_mosquito_model, SimpleSampler, get_simulator


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


def test_hybrid_central_noncentral_model():
    '''
    '''
    # (sample, log_prob, reconstruct_state, sample_diffs), (real_params, n_states) = simple_hybrid_model()

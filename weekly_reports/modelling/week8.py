'''
Working on estimation of parameters in compartementalized state space models using Hamiltonian Monte Carlo.
Starting with a simple SEIR model, we will use the NUTS algorithm to estimate the parameters of the model.

Goals for this week:
- Run the model with ClimateHealthTimeSeries data
- Inlcude Mosquito compartments in the model
- Use data from multiple locations
- Use weather data on finer resolution than health data (i.e. daily weather data and monthly health data)
'''

import numpy as np
import jax
from probabilistic_machine_learning.cases.diff_encoded_mosquito import diff_encoded_model, simple_model, \
    model_evaluation_plot, debug_logprob, advancing_model
from probabilistic_machine_learning.adaptors.jax_nuts import sample as nuts_sample
import matplotlib.pyplot as plt
from report_tests import get_markdown
from climate_health.datatypes import ClimateData, HealthData, ClimateHealthTimeSeries
import plotly.express as px

from climate_health.plotting import plot_timeseries_data
from climate_health.plotting.prediction_plot import prediction_plot


def test_estimate_single_parameter():
    '''Samples data from a simple SEIR model and estimates the beta parameter using NUTS. Includes a time varying temperature parameter.'''
    real_params = {'beta': 0.5}
    sample, log_prob, reconstruct_state = simple_model()
    T = 100
    temperature = np.random.normal(1, 1, T - 1)
    observed = sample(T, jax.random.PRNGKey(100), real_params, temperature)
    init_diffs = np.random.normal(0, 1, (T - 1))
    init_dict = {'logits_array': init_diffs} | {'beta': 0.0}
    lp = log_prob(observed, temperature)
    samples = nuts_sample(lp, jax.random.PRNGKey(0),
                          init_dict, 100, 100)

    model_evaluation_plot(sample, real_params, samples, temperature)
    return plt.gcf()


def test_run_with_climate_health_data():
    '''Runs the model with ClimateHealthTimeSeries data
    - Load climate data from file
    - Simulate health data using model
    - Run model on climatehealth data set
    - Plot model evaluation
    '''
    climate_data = ClimateData.from_csv('../../example_data/climate_data.csv')
    T = len(climate_data) + 1
    sample, log_prob, reconstruct_state = simple_model()
    real_params = {'beta': 0.001}

    class Simulator:
        def simulate(self, climate_data: ClimateData):
            samples = sample(T, jax.random.PRNGKey(0), real_params, climate_data.max_temperature)
            return HealthData(time_period=climate_data.time_period, disease_cases=samples)

    health_data = Simulator().simulate(climate_data)
    init_diffs = np.random.normal(0, 1, (T - 1))
    init_dict = {'logits_array': init_diffs} | {'beta': 0.0}
    lp = log_prob(health_data.disease_cases, climate_data.max_temperature)
    samples = nuts_sample(lp, jax.random.PRNGKey(0),
                          init_dict, 50, 50)
    model_evaluation_plot(sample, real_params, samples, climate_data.max_temperature)
    return plt.gcf()


def get_simulator(sample, real_params):
    class Simulator:
        def simulate(self, climate_data: ClimateData):
            T = len(climate_data) + 1
            samples = sample(T, jax.random.PRNGKey(0), real_params, climate_data.max_temperature)
            return HealthData(time_period=climate_data.time_period, disease_cases=samples)

    simulator = Simulator()
    return simulator


class SimpleSampler:
    def __init__(self, key, log_prob: callable, sample_func: callable, param_names: list[str], n_states=None):
        self._log_prob = log_prob
        self._param_names = param_names
        self._param_samples = None
        self._sample_key = key
        self._sample_func = sample_func
        self._n_states = n_states

    @property
    def n_samples(self):
        return len(self._param_samples[list(self._param_samples)[0]])

    def train(self, time_series: ClimateHealthTimeSeries) -> HealthData:
        T = len(time_series) + 1
        if self._n_states is None:
            init_diffs = np.random.normal(0, 1, (T - 1))
        else:
            init_diffs = np.random.normal(0, 1, (T - 1, self._n_states)).reshape((T - 1, self._n_states))
        init_dict = {'logits_array': init_diffs} | {param_name: np.random.normal(0, 10) for param_name in
                                                    self._param_names}
        lp = self._log_prob(time_series.disease_cases, time_series.mean_temperature)
        self._sample_key, key = jax.random.split(self._sample_key)
        self._param_samples = nuts_sample(lp, key, init_dict, 50, 500)

    def sample(self, climate_data: ClimateData) -> HealthData:
        T = len(climate_data) + 1
        self._sample_key, key1, key2 = jax.random.split(self._sample_key, 3)

        param_nr = jax.random.randint(key2, (1,), 0, self.n_samples)[0]
        params = {param_name: self._param_samples[param_name][param_nr] for param_name in self._param_names}
        return self._sample_func(T, key1, params, climate_data.mean_temperature)


def test_simplified_interface():
    '''
    Automate boilerplate code
    '''

    model = lambda: (simple_model(), (['beta'], None))
    figure = check_model_capacity(model)
    return figure.show()


def test_mored_advanced_model():
    '''
    Check model_capacity for full SEIR model
    '''
    model = advancing_model
    return check_model_capacity(model).show()


def check_model_capacity(model):
    (sample, log_prob, reconstruct_state), (param_names, n_states) = model()
    real_params = {name: np.random.normal(0, 10) for name in param_names}
    climate_data = ClimateData.from_csv('../../example_data/climate_data.csv')
    simulator = get_simulator(sample, real_params)
    health_data = simulator.simulate(climate_data)
    sampler = SimpleSampler(jax.random.PRNGKey(0), log_prob, sample, param_names, n_states)
    data_set = ClimateHealthTimeSeries.combine(health_data, climate_data)
    sampler.train(data_set)
    figure = prediction_plot(health_data, sampler, climate_data, 10)
    return figure


if __name__ == '__main__':
    mds = [get_markdown(f) for f in (estimate_single_parameter, run_with_climate_health_data)]
    print(mds)
    md = '\n'.join(mds)
    with open('week8.md', 'w') as f:
        f.write(md)

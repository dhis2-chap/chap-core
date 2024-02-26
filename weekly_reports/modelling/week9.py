'''
Should get the mosquito model incorporated into the human model.
 With that and different scale for health and climate data, it is ready to use on real data.

- Inlcude Mosquito compartments in the model
- Use data from multiple locations
- Use weather data on finer resolution than health data (i.e. daily weather data and monthly health data)
'''
import numpy as np
from matplotlib import pyplot as plt
from report_tests import show

import jax
from probabilistic_machine_learning.cases.diff_encoded_mosquito import pure_mosquito_model, full_model
from scipy.special import logit, expit

from climate_health.datatypes import ClimateHealthTimeSeries, ClimateData
from climate_health.plotting.prediction_plot import prediction_plot
from tests import EXAMPLE_DATA_PATH
from .week8 import check_model_capacity, get_parameterized_mosquito_model, SimpleSampler, get_simulator


def test_human_mosquito_model():
    '''Find a good parameter space for human/mosquito model'''

    (sample, log_prob, reconstruct_state), (param_names, n_states) = full_model()
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
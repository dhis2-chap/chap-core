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
    model_evaluation_plot, debug_logprob
from probabilistic_machine_learning.adaptors.jax_nuts import sample as nuts_sample
import matplotlib.pyplot as plt
from report_tests import get_markdown

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
                          init_dict, 50, 50)

    model_evaluation_plot(sample, real_params, samples, temperature)
    return plt.gcf()

def test_run_with_climate_health_data():
    '''Runs the model with ClimateHealthTimeSeries data
    - Load climate data from file
    - Simulate health data using model
    - Run model on climatehealth data set
    - Plot model evaluation
    '''

    

md = get_markdown(test_estimate_single_parameter)
with open('week8.md', 'w') as f:
    f.write(md)

'''
Should get the mosquito model incorporated into the human model.
 With that and different scale for health and climate data, it is ready to use on real data.

- Inlcude Mosquito compartments in the model
- Use data from multiple locations
- Use weather data on finer resolution than health data (i.e. daily weather data and monthly health data)
'''
from report_tests import show

import jax
from probabilistic_machine_learning.cases.diff_encoded_mosquito import pure_mosquito_model

from climate_health.datatypes import ClimateHealthTimeSeries
from climate_health.plotting.prediction_plot import prediction_plot
from .week8 import check_model_capacity, get_parameterized_mosquito_model, SimpleSampler


def test_human_mosquito_model():
    mosquito_data, climate_data, real_params = get_parameterized_mosquito_model()
    (sample, log_prob, reconstruct_state), (param_names, n_states) = pure_mosquito_model()
    health_data = mosquito_data
    sampler = SimpleSampler(jax.random.PRNGKey(0),
                            log_prob, sample,
                            param_names, n_states,
                            n_warmup_samples=500)
    data_set = ClimateHealthTimeSeries.combine(health_data, climate_data)
    sampler.train(data_set, init_values=real_params)
    fig = prediction_plot(health_data, sampler, climate_data, 10)
    return show(fig)
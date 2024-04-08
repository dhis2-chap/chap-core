from functools import partial

from climate_health.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from .hmc import sample
from .jax import jax, PRNGKey, stats, jnp
from .regression_model import remove_nans
import logging

logger = logging.getLogger(__name__)


class SSM:

    def __init__(self):
        self._initial_params = {'infected_decay': 0.9, 'beta_temp': 0.1}

    def _log_infected_dist(self, params, prev_log_infected, mean_temp):
        mu = prev_log_infected * params['infected_decay'] + params['beta_temp'] * mean_temp
        return partial(stats.norm.logpdf, loc=mu, scale=1)

    def train(self, data: SpatioTemporalDict[ClimateHealthTimeSeries]):
        logger.info(f"Training model with {len(data.locations())} locations")
        data = {name: remove_nans(data.data())
                for name, data in data.items()}
        init_params = self._initial_params
        init_params['log_infected'] = {location: jnp.log(data.disease_cases)
                                       for location, data in data.items()}
        init_params['log_observation_rate'] = {location: jnp.log(0.1) for location in data.keys()}

        def logpdf(params):
            obs_pdf = []
            state_pdf = []
            for location, local_data in data.items():
                rate = jnp.exp(params['log_infected'][location] + params['log_observation_rate'][location])
                obs_pdf.append(
                    stats.poisson.logpmf(local_data.disease_cases, rate).sum())
                infected_dist = self._log_infected_dist(params, params['log_infected'][location][:-1],
                                                        local_data.mean_temperature[:-1])
                infected_pdf = infected_dist(params['log_infected'][location][1:]).sum()

                state_pdf.append(infected_pdf)

            return sum(obs_pdf)

        logpdf(self._initial_params)
        self._sampled_params = sample(logpdf, PRNGKey(0), init_params, 10, 10)
        print(self._sampled_params)
        last_pdf = logpdf(extract_last(self._sampled_params))
        assert not jnp.isnan(last_pdf)

    def predict(self, data: SpatioTemporalDict[ClimateData]):
        last_params = extract_last(self._sampled_params)
        predictions = {}
        for location, local_data in data.items():
            time_period = data.get_location(location).data().time_period[:1]
            log_infected = last_params['log_infected'][location]
            log_observation_rate = last_params['log_observation_rate'][location]
            rate = jnp.exp(log_infected[-1:] + log_observation_rate)
            predictions[location] = HealthData(time_period, jnp.exp(rate))

        return SpatioTemporalDict(predictions)


def extract_last(samples):
    return {key: value[-1] if not hasattr(value, 'items') else extract_last(value) for key, value in samples.items()}

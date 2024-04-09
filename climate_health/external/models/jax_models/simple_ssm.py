from functools import partial
from .util import extract_last
from climate_health.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from .hmc import sample
from .jax import jax, PRNGKey, stats, jnp
from .regression_model import remove_nans
import logging

logger = logging.getLogger(__name__)


class NaiveSSM:
    def __init__(self):
        self._sampled_params = None
        self._initial_params = {'infected_decay': 0.9, 'alpha': 0.1}
        self._prior = self._get_priors(self._initial_params.keys())

    def _get_priors(self, param_names):
        return {name: partial(stats.norm.logpdf, loc=0.1, scale=1) for name in param_names}

    def _log_infected_dist(self, params, prev_log_infected, mean_temp):
        mu = prev_log_infected * params['infected_decay'] + self._temperature_effect(mean_temp, params)
        return partial(stats.norm.logpdf, loc=mu, scale=1)

    def _temperature_effect(self, mean_temp, params):
        return params['alpha']

    def train(self, data: SpatioTemporalDict[ClimateHealthTimeSeries]):
        logger.info(f"Training model with {len(data.locations())} locations")
        data = {name: remove_nans(data.data())
                for name, data in data.items()}
        init_params = self._initial_params
        init_params['log_infected'] = {location: jnp.log(data.disease_cases + 1)
                                       for location, data in data.items()}
        init_params['log_observation_rate'] = {location: jnp.log(0.1) for location in data.keys()}
        prior_pdf_func = lambda params: sum([prior(params[name]) for name, prior in self._prior.items()])

        def obs_pdf_func(params, local_data, location):
            rate = jnp.exp(params['log_infected'][location] + params['log_observation_rate'][location])
            obs_p = stats.poisson.logpmf(local_data.disease_cases, rate).sum()
            return obs_p

        def infected_pdf_func(params, local_data, location):
            infected_dist = self._log_infected_dist(params, params['log_infected'][location][:-1],
                                                    local_data.mean_temperature[:-1])
            return infected_dist(params['log_infected'][location][1:]).sum()

        def logpdf(params):
            obs_pdf = []
            state_pdf = []
            prior_pdf = prior_pdf_func(params)
            for location, local_data in data.items():
                obs_p = obs_pdf_func(params, local_data, location)
                obs_pdf.append(obs_p)
                infected_pdf = infected_pdf_func(params, local_data, location)
                state_pdf.append(infected_pdf)

            return sum(obs_pdf) + sum(state_pdf) + prior_pdf

        first_pdf = logpdf(init_params)
        if jnp.isnan(first_pdf):
            print(prior_pdf_func(init_params))
            print([obs_pdf_func(init_params, local_data, location) for location, local_data in data.items()])
            print([infected_pdf_func(init_params, local_data, location) for location, local_data in data.items()])
        assert not jnp.isnan(first_pdf)
        print('first_pdf', first_pdf)
        first_grad = jax.grad(logpdf)(init_params)
        print('first_grad', first_grad)
        self._sampled_params = sample(logpdf, PRNGKey(0), init_params, 10, 100)
        last_params = extract_last(self._sampled_params)
        last_pdf = logpdf(last_params)
        print(last_pdf)
        print(last_params)
        assert not jnp.isnan(last_pdf)

    def predict(self, data: SpatioTemporalDict[ClimateData]):
        last_params = extract_last(self._sampled_params)
        predictions = {}
        for location, local_data in data.items():
            time_period = data.get_location(location).data().time_period[:1]
            log_infected = last_params['log_infected'][location]
            log_observation_rate = last_params['log_observation_rate'][location]
            rate = jnp.exp(log_infected[-1:] + log_observation_rate)
            predictions[location] = HealthData(time_period, rate)

        return SpatioTemporalDict(predictions)


class SeasonalSSM(NaiveSSM):
    def __init__(self):
        super().__init__()
        self._initial_params['seasonal_effect'] = jnp.zeros(12)
        self._prior['seasonal_effect'] = partial(stats.norm.logpdf, loc=0, scale=1)

    def _temperature_effect(self, mean_temp, params):
        return params['alpha'] + jnp.sum(params['seasonal_effect'][mean_temp.month - 1])


class SSM(NaiveSSM):
    def __init__(self):
        self._initial_params = {'infected_decay': 0.9, 'beta_temp': 0.1}
        self._prior = self._get_priors(self._initial_params.keys())

    def _temperature_effect(self, mean_temp, params):
        return params['beta_temp'] * mean_temp


class SSMWithLinearEffect(SSM):
    ''' SSM where the temperature effect is modelled as a sigmoid function of temperature. '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_params['alpha_temp'] = 0.1
        self._prior['alpha_temp'] = partial(stats.norm.logpdf, loc=0.1, scale=0.1)

    def _temperature_effect(self, mean_temp, params):
        return params['beta_temp'] * mean_temp + params['alpha_temp']


class SSMWithSigmoidEffect(SSM):
    ''' SSM where the temperature effect is modelled as a sigmoid function of temperature. '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        param_names = ['a_temp', 'b_temp', 'c_temp']
        self._initial_params.update({name: 0.1 for name in param_names})
        self._prior.update({name: partial(stats.norm.logpdf, loc=0.1, scale=1) for name in param_names})

    def _temperature_effect(self, mean_temp, params):
        return jax.scipy.special.expit(params['beta_temp'] * mean_temp + params['a_temp']) * params['b_temp'] + params[
            'c_temp']

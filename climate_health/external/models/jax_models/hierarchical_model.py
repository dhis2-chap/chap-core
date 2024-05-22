import dataclasses
from collections import defaultdict
from typing import Any, Optional, Callable
import numpy as np

from climate_health.time_period.date_util_wrapper import delta_month
from .deterministic_seir_model import MarkovChain
from .hmc import sample
from .jax import jax, PRNGKey, jnp
from bionumpy.bnpdataclass import BNPDataClass, bnpdataclass

from climate_health.datatypes import ClimateHealthTimeSeries, HealthData, ClimateData, FullData, SummaryStatistics
from climate_health.external.models.jax_models.prototype_hierarchical import hierarchical_linear_regression, \
    GlobalSeasonalParams, DistrictParams, seasonal_linear_regression, get_hierarchy_logprob_func, \
    join_global_and_district, hierarchical
from climate_health.external.models.jax_models.utii import get_state_transform, state_or_param, tree_sample, index_tree
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from .model_spec import Poisson, PoissonSkipNaN, Normal
from .protoype_annotated_spec import Positive
from .simple_ssm import get_summary
from .util import array_tree_length


@hierarchical('District')
class SeasonalDistrictParams(DistrictParams):
    alpha: float = 0.
    beta: float = 0.
    month_effect: np.ndarray = tuple((0.,)) * 12

@bnpdataclass
class SeasonalClimateHealthData(FullData):
    month: int
    year: int

@bnpdataclass
class SeasonalClimateHealthDataState(SeasonalClimateHealthData):
    time_index: int


def create_seasonal_data(data: BNPDataClass):
    @bnpdataclass
    class SeasonalData(data.__class__):
        month: int
        year: int
        time_index: int

    months = [period.month for period in data.time_period]
    years = [period.year for period in data.time_period]
    time_index = [period.year*12 + period.month for period in data.time_period]

    return SeasonalData(
        **{field.name: getattr(data, field.name) for field in dataclasses.fields(data)},
        month=months, year=years, time_index=time_index)


class HierarchicalModel:
    def __init__(self, key: PRNGKey = PRNGKey(0), params: Optional[dict[str, Any]] = None, num_samples: int = 100,
                 num_warmup: int = 100):
        self.params = params
        self._key = key
        self._regression_model = None
        self._min_year = None
        self._max_year = None
        self._num_samples = num_samples
        self._num_warmup = num_warmup
        self._regression_model: Optional[Callable] = None
        self._standardization_func = lambda x: x

    def _get_standardization_func(self, data: SpatioTemporalDict[ClimateHealthTimeSeries]):
        values = np.concatenate([value.mean_temperature for value in data.values()])
        mean = np.mean(values)
        std = np.std(values)
        return lambda x: (x - mean) / std

    def _set_model(self, data_dict: SpatioTemporalDict[SeasonalClimateHealthData]):
        min_year = min([min(value.year) for value in data_dict.values()])
        max_year = max([max(value.year) for value in data_dict.values()])
        n_years = max_year - min_year + 1


        @state_or_param
        class ParamClass(GlobalSeasonalParams):
            observation_rate: Positive = 0.01
            year_effect: np.ndarray = tuple((0.,)) * n_years

        self._param_class = ParamClass

        def ch_regression(params: 'ParamClass', given: SeasonalClimateHealthData) -> HealthData:
            log_rate = params.alpha + params.beta * self._standardization_func(given.mean_temperature) + params.month_effect[given.month - 1] + \
                       params.year_effect[given.year - min_year]
            final_rate = jnp.exp(log_rate) * given.population * params.observation_rate + 0.1
            return PoissonSkipNaN(final_rate)

        self._regression_model = ch_regression


    def train(self, data: SpatioTemporalDict[FullData]):
        random_key, self._key = jax.random.split(self._key)
        data_dict = {key: create_seasonal_data(value.data()) for key, value in data.items()}
        self._set_model(data_dict)
        T_Param, transform, inv = get_state_transform(self._param_class)
        T_ParamD, transformD, invD = get_state_transform(SeasonalDistrictParams)
        logprob_func = get_hierarchy_logprob_func(
            self._param_class, SeasonalDistrictParams, data_dict,
            self._regression_model, observed_name='disease_cases')

        init_params = T_Param().sample(random_key), {location: T_ParamD().sample(random_key) for location in data_dict.keys()}
        val = logprob_func(init_params)
        assert not jnp.isnan(val), val
        assert not jnp.isinf(val), val
        grad = jax.grad(logprob_func)(init_params)
        assert not jnp.isnan(grad[0].alpha), grad
        raw_samples = sample(logprob_func, random_key, init_params, num_samples=self._num_warmup, num_warmup=self._num_warmup)
        self.params = (transform(raw_samples[0]), {name: transformD(sample) for name, sample in raw_samples[1].items()})
        last_params = index_tree(raw_samples, -1)
        assert not jnp.isinf(logprob_func(last_params)), logprob_func(last_params)
        assert not jnp.isnan(jax.grad(logprob_func)(last_params)[0].alpha), jax.grad(logprob_func)(last_params)

    def sample(self, data: SpatioTemporalDict[ClimateData], n=1) -> SpatioTemporalDict[HealthData]:
        params = index_tree(self.params, -1)
        random_key, self._key = jax.random.split(self._key)
        data_dict = {key: create_seasonal_data(value.data()) for key, value in data.items()}
        true_params = {name: join_global_and_district(params[0],
                                                      params[1][name])
                       for name in data_dict.keys()}
        return SpatioTemporalDict({key: self._regression_model(true_params[key], data_dict[key]).sample(random_key)
                                   for key in data_dict.keys()})
    def _adapt_params(self, params, data_dict):
        return params

    def forecast(self, future_weather: SpatioTemporalDict[ClimateData], n_samples=1000,
                 forecast_delta=6 * delta_month) -> SpatioTemporalDict[SummaryStatistics]:
        time_period = next(iter(future_weather.data())).data().time_period
        n_periods = forecast_delta // time_period.delta
        time_period = time_period[:n_periods]
        future_weather = SpatioTemporalDict({key: value.data()[:n_periods] for key, value in future_weather.items()})
        num_samples = n_samples
        param_key, self._key = jax.random.split(self._key)
        n_sampled_params = array_tree_length(self.params)
        param_idxs = jax.random.randint(param_key, (num_samples,), 0, n_sampled_params)
        samples = defaultdict(list)
        data_dict = {key: create_seasonal_data(value.data()) for key, value in future_weather.items()}
        for param_idx, random_key in zip(param_idxs, jax.random.split(self._key, num_samples)):
            params = index_tree(self.params, param_idx)
            params = self._adapt_params(params, data_dict)
            true_params = {name: join_global_and_district(params[0],
                                                          params[1][name])
                           for name in data_dict.keys()}
            for key, value in data_dict.items():
                new_key, random_key = jax.random.split(random_key)
                samples[key].append(self._regression_model(true_params[key], value).sample(new_key))
        return SpatioTemporalDict(
            {key: get_summary(time_period, np.array(value)) for key, value in samples.items()})

    def predict(self, *args, **kwargs):
        return self.forecast(*args, **kwargs)


class HierarchicalStateModel(HierarchicalModel):
    def _adapt_params(self, params_tuple, data_dict):
        max_idx = max([max(value.time_index) for value in data_dict.values()])
        if max_idx<=self._idx_range[1]:
            return params_tuple
        params, district_params = params_tuple
        sigma=0.5
        new_markov_chain = MarkovChain(self._transition,
                                       Normal(params.state[-1], sigma),
                                       np.arange(self._idx_range[1] + 1, max_idx + 1))
        key, self._key = jax.random.split(self._key)
        return dataclasses.replace(params, state=np.concatenate([params.state, new_markov_chain.sample(key)])), district_params


    def _set_model(self, data_dict: SpatioTemporalDict[SeasonalClimateHealthDataState]):
        min_idx = min([min(value.time_index) for value in data_dict.values()])
        max_idx = max([max(value.time_index) for value in data_dict.values()])
        self._idx_range = (min_idx, max_idx)
        sigma = 0.5
        self._transition = lambda state: Normal(state, sigma)

        @state_or_param
        class ParamClass(GlobalSeasonalParams):
            observation_rate: Positive = 0.01
            state: MarkovChain = MarkovChain(
                self._transition,
                Normal(0., sigma),
                np.arange(min_idx, max_idx + 1))

        self._param_class = ParamClass
        self._standardization_func = self._get_standardization_func(data_dict)

        def ch_regression(params: 'ParamClass', given: SeasonalClimateHealthData) -> HealthData:
            idx = given.time_index - min_idx
            log_rate = params.alpha + params.month_effect[given.month - 1] + params.state[idx]
            final_rate = jnp.exp(log_rate) * given.population * params.observation_rate + 0.1
            return PoissonSkipNaN(final_rate)

        self._regression_model = ch_regression

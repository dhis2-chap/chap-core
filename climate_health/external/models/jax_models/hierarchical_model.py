import plotly.express as px
import dataclasses
from collections import defaultdict
from functools import partial
from typing import Any, Optional, Callable
import numpy as np

from climate_health.time_period.date_util_wrapper import delta_month
from .deterministic_seir_model import MarkovChain
from .hmc import sample
from .jax import jax, PRNGKey, jnp, expit, logit
from bionumpy.bnpdataclass import BNPDataClass, bnpdataclass

from climate_health.datatypes import ClimateHealthTimeSeries, HealthData, ClimateData, FullData, SummaryStatistics
from climate_health.external.models.jax_models.prototype_hierarchical import hierarchical_linear_regression, \
    GlobalSeasonalParams, DistrictParams, seasonal_linear_regression, get_hierarchy_logprob_func, \
    join_global_and_district, hierarchical, HierarchyLogProbFunc, HiearchicalLogProbFuncWithStates, \
    HiearchicalLogProbFuncWithDistrictStates
from climate_health.external.models.jax_models.utii import get_state_transform, state_or_param, tree_sample, index_tree, \
    PydanticTree
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from .model_spec import Poisson, PoissonSkipNaN, Normal, distributionclass
from .protoype_annotated_spec import Positive
from .simple_ssm import get_summary
from .util import array_tree_length

SIGMA = 0.5


def get_state_dist_from_params(state_params, n_periods):
    sigma = SIGMA  # state_params.sigma
    return MarkovChain(lambda state: Normal(state, sigma),
                       Normal(state_params.init, sigma),
                       np.arange(n_periods))


def get_state_regression_dist_from_params(state_params, n_periods):
    sigma = SIGMA  # state_params.sigma

    def mean_func(mean):
        p = expit(mean)
        return logit(p + p * (1 - p) * state_params.beta - p * state_params.gamma + 1e-6)

    return MarkovChain(lambda state: Normal(mean_func(state), sigma),
                       Normal(state_params.init, sigma * 3),
                       np.arange(n_periods))


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
    time_index = [period.year * 12 + period.month for period in data.time_period]

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
            log_rate = params.alpha + params.beta * self._standardization_func(given.mean_temperature) + \
                       params.month_effect[given.month - 1] + \
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
        logprob_func = self._get_log_prob_func(data_dict)
        init_params = T_Param().sample(random_key), {location: T_ParamD().sample(random_key) for location in
                                                     data_dict.keys()}
        init_params = self._add_init_params(init_params)
        val = logprob_func(init_params)
        assert not jnp.isnan(val), val
        assert not jnp.isinf(val), val
        grad = jax.grad(logprob_func)(init_params)
        assert not jnp.isnan(grad[0].alpha), grad
        raw_samples = sample(logprob_func, random_key, init_params, num_samples=self._num_warmup,
                             num_warmup=self._num_warmup)
        self.save_params(raw_samples, transform, transformD)
        last_params = index_tree(raw_samples, -1)
        assert not jnp.isinf(logprob_func(last_params)), logprob_func(last_params)
        assert not jnp.isnan(jax.grad(logprob_func)(last_params)[0].alpha), jax.grad(logprob_func)(last_params)

    def save_params(self, raw_samples, transform, transformD):
        self.params = (transform(raw_samples[0]), {name: transformD(sample) for name, sample in raw_samples[1].items()})

    def _get_log_prob_func(self, data_dict):
        return HierarchyLogProbFunc(
            self._param_class, SeasonalDistrictParams, data_dict,
            self._regression_model, observed_name='disease_cases')

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
                samples[key].append(self._sample_from_model(key, new_key, params, true_params, value))
        return SpatioTemporalDict(
            {key: get_summary(time_period, np.array(value)) for key, value in samples.items()})

    def _sample_from_model(self, key, new_key, params, true_params, value):
        return self._regression_model(true_params[key], value, *params[2:]).sample(new_key)

    def predict(self, *args, **kwargs):
        return self.forecast(*args, **kwargs)

    def _add_init_params(self, init_params):
        return init_params


@state_or_param
class StateParams(PydanticTree):
    init: float = 0.


@state_or_param
class RegressionStateParams(StateParams):
    beta: Positive = 0.1
    gamma: Positive = 0.1


@state_or_param
class GlobalParams(GlobalSeasonalParams):
    observation_rate: Positive = 0.01
    state_params: StateParams = StateParams(0.)  # , sigma)
    district_state_params: StateParams = StateParams(0.)


@state_or_param
class GlobalParams2(GlobalSeasonalParams):
    observation_rate: Positive = 0.01
    state_params: RegressionStateParams = RegressionStateParams(0., 0.1, 0.1)  # , sigma)
    district_state_params: RegressionStateParams = RegressionStateParams(0., 0.1, 0.1)  # , sigma)


class HierarchicalStateModel(HierarchicalModel):
    _log_prog_func_class = HiearchicalLogProbFuncWithStates
    _param_class = GlobalParams

    @property
    def _state_dist_func(self):
        return get_state_dist_from_params

    @property
    def _global_state_dist_func(self):
        return get_state_dist_from_params

    def _adapt_params(self, params_tuple, data_dict):
        max_idx = max([max(value.time_index) for value in data_dict.values()])
        if max_idx <= self._idx_range[1]:
            return params_tuple
        params, district_params, states = params_tuple
        prediction_params = StateParams(states[-1])  # , params.sigma)
        new_markov_chain = self._state_dist_func(prediction_params, len(np.arange(self._idx_range[1] + 1, max_idx + 1)))
        key, self._key = jax.random.split(self._key)
        return params, district_params, np.concatenate([states, new_markov_chain.sample(key)])

    def _add_init_params(self, init_params):
        init_states = np.zeros(self._idx_range[1] + 1 - self._idx_range[0])
        return init_params + (init_states,)

    def save_params(self, raw_samples, transform, transformD):
        self.params = (transform(raw_samples[0]),
                       {name: transformD(sample) for name, sample in raw_samples[1].items()},
                       raw_samples[2])

    def diagnose(self):
        flat_tree = jax.tree_util.tree_flatten(self.params)
        for val in flat_tree[0]:
            if val.ndim == 1:
                px.line(val).show()

    def _set_model(self, data_dict: SpatioTemporalDict[SeasonalClimateHealthDataState]):
        min_idx = min([min(value.time_index) for value in data_dict.values()])
        max_idx = max([max(value.time_index) for value in data_dict.values()])
        self._idx_range = (min_idx, max_idx)
        sigma = SIGMA
        self._transition = lambda state: Normal(state, sigma)

        self._standardization_func = self._get_standardization_func(data_dict)

        def ch_regression(params: 'ParamClass', given: SeasonalClimateHealthData, state) -> HealthData:
            idx = given.time_index - min_idx
            log_rate = params.alpha + params.month_effect[given.month - 1] + state[idx]
            final_rate = expit(log_rate) * given.population * params.observation_rate + 0.1
            return PoissonSkipNaN(final_rate)

        self._regression_model = ch_regression

    def _get_log_prob_func(self, data_dict):
        return self._log_prog_func_class(
            self._param_class, SeasonalDistrictParams, data_dict,
            self._regression_model, observed_name='disease_cases',
            state_class=partial(self._state_dist_func,
                                n_periods=self._idx_range[1] + 1 - self._idx_range[0]))


class HierarchicalStateModelD(HierarchicalStateModel):
    _log_prog_func_class = HiearchicalLogProbFuncWithDistrictStates

    def _adapt_params(self, params_tuple, data_dict):
        ''' This needs to automatically generated from the model'''
        max_idx = max([max(value.time_index) for value in data_dict.values()])
        if max_idx <= self._idx_range[1]:
            return params_tuple
        params, district_params, (global_states, states_dict) = params_tuple
        d = {}
        for name, states in states_dict.items():
            prediction_params = dataclasses.replace(params.district_state_params, init=states[-1])
            new_markov_chain = self._state_dist_func(prediction_params,
                                                     len(np.arange(self._idx_range[1] + 1, max_idx + 1)))
            key, self._key = jax.random.split(self._key)
            d[name] = np.concatenate([states, new_markov_chain.sample(key)])
        prediction_params = dataclasses.replace(params.state_params, init=global_states[-1])
        new_markov_chain = self._state_dist_func(prediction_params, len(np.arange(self._idx_range[1] + 1, max_idx + 1)))
        key, self._key = jax.random.split(self._key)
        global_new_states = np.concatenate([global_states, new_markov_chain.sample(key)])
        return params, district_params, (global_new_states, d)

    def _sample_from_model(self, key, new_key, params, true_params, value):
        return self._regression_model(true_params[key], value, params[2][1][key] + params[2][0]).sample(new_key)

    def _add_init_params(self, init_params):
        init_states = {name: np.zeros(self._idx_range[1] + 1 - self._idx_range[0])
                       for name in init_params[1]}
        init_global_state = np.zeros(self._idx_range[1] + 1 - self._idx_range[0])
        return init_params + ((init_global_state, init_states),)


class HierarchicalStateModelD2(HierarchicalStateModelD):
    _param_class = GlobalParams2

    @property
    def _state_dist_func(self):
        return get_state_regression_dist_from_params


class _SimpleMarkovChain:
    def __init__(self, state_params):
        self._chain = MarkovChain(lambda state: Normal(state, state_params.sigma),
                                  Normal(state_params.init, state_params.sigma),
                                  np.arange(state_params.n))

    def log_prob(self, *args, **kwargs):
        return self._chain.log_prob(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self._chain.sample(*args, **kwargs)

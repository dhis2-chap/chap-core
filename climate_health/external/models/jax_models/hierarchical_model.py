import dataclasses
from typing import Any, Optional, Callable

import numpy as np

from .hmc import sample
from .jax import jax, PRNGKey, jnp
from bionumpy.bnpdataclass import BNPDataClass, bnpdataclass

from climate_health.datatypes import ClimateHealthTimeSeries, HealthData, ClimateData, FullData
from climate_health.external.models.jax_models.prototype_hierarchical import hierarchical_linear_regression, \
    GlobalSeasonalParams, DistrictParams, seasonal_linear_regression, get_hierarchy_logprob_func, \
    join_global_and_district
from climate_health.external.models.jax_models.utii import get_state_transform, state_or_param, tree_sample, index_tree
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from .model_spec import Poisson, PoissonSkipNaN
from .protoype_annotated_spec import Positive


@bnpdataclass
class SeasonalClimateHealthData(FullData):
    month: int
    year: int


def create_seasonal_data(data: BNPDataClass):
    @bnpdataclass
    class SeasonalData(data.__class__):
        month: int
        year: int

    months = [period.month for period in data.time_period]
    years = [period.year for period in data.time_period]
    print(data)
    return SeasonalData(
        **{field.name: getattr(data, field.name) for field in dataclasses.fields(data)},
        month=months, year=years)


class HierarchicalModel:
    def __init__(self, key: PRNGKey, params: Optional[dict[str, Any]] = None, num_samples: int = 100, num_warmup: int = 100):
        self.params = params
        self._key = key
        self._regression_model = None
        self._min_year = None
        self._max_year = None
        self._num_samples = num_samples
        self._num_warmup = num_warmup
        self._regression_model: Optional[Callable]=None

    def train(self, data: SpatioTemporalDict[FullData]):
        random_key, self._key = jax.random.split(self._key)
        data_dict = {key: create_seasonal_data(value.data()) for key, value in data.items()}
        min_year =  min([min(value.year) for value in data_dict.values()])
        max_year = max([max(value.year) for value in data_dict.values()])
        n_years = max_year - min_year + 1

        @state_or_param
        class ParamClass(GlobalSeasonalParams):
            observation_rate: Positive = 0.01
            year_effect: np.ndarray = tuple((0.,))*n_years

        self._param_class = ParamClass

        T_Param, transform, inv = get_state_transform(ParamClass)
        T_ParamD, transformD, invD = get_state_transform(DistrictParams)

        def ch_regression(params: ParamClass, given: SeasonalClimateHealthData) -> HealthData:
            log_rate = params.alpha + params.beta * given.mean_temperature + params.month_effect[given.month-1] + params.year_effect[given.year-min_year]
            final_rate = jnp.exp(log_rate) * given.population * params.observation_rate + 0.1
            return PoissonSkipNaN(final_rate)

        self._regression_model = ch_regression
        logprob_func = get_hierarchy_logprob_func(
            ParamClass, DistrictParams, data_dict,
            ch_regression, observed_name='disease_cases')
        init_params = T_Param().sample(random_key), {location: T_ParamD().sample(random_key) for location in data_dict.keys()}
        init_params = T_Param(0., 0., 0., np.zeros(12), np.log(0.01), np.zeros(n_years)), {name: T_ParamD(0., 0.) for
                                                                                           name in data_dict.keys()}
        val = logprob_func(init_params)
        print('Value: ', val)
        grad = jax.grad(logprob_func)(init_params)
        assert not jnp.isnan(grad[0].alpha), grad
        print('Grad: ', grad)
        raw_samples = sample(logprob_func, random_key, init_params,
                             num_samples=self._num_warmup, num_warmup=self._num_warmup)
        self.params = (transform(raw_samples[0]), {name: transformD(sample) for name, sample in raw_samples[1].items()})
        last_params = index_tree(raw_samples, -1)
        print(last_params)
        assert not jnp.isinf(logprob_func(last_params)), logprob_func(last_params)
        assert not jnp.isnan(jax.grad(logprob_func)(last_params)[0].alpha), jax.grad(logprob_func)(last_params)

    def sample(self, data: SpatioTemporalDict[ClimateData]) -> SpatioTemporalDict[HealthData]:
        random_key, self._key = jax.random.split(self._key)
        data_dict = {key: create_seasonal_data(value.data()) for key, value in data.items()}
        last_params = index_tree(self.params, -1)
        model = hierarchical_linear_regression(*last_params, data_dict, self._regression_model)
        samples = tree_sample(model, self._key)
        T_Param, transform, inv = get_state_transform(self._param_class)
        T_ParamD, transformD, invD = get_state_transform(DistrictParams)
        logprob_func = get_hierarchy_logprob_func(
            self._param_class, DistrictParams, data_dict,
            self._regression_model, observed_name='disease_cases')
        true_params = {name: join_global_and_district(transform(last_params[0]),
                                                      transformD(last_params[1][name]))
                       for name in data_dict.keys()}
        return SpatioTemporalDict({key: self._regression_model(true_params[key], data_dict[key]).sample(random_key)
                                   for key in data_dict.keys()})

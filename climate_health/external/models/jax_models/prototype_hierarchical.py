import dataclasses
from dataclasses import replace
from typing import Sequence, Callable, Any
from .jax import expit, logit

import numpy as np
from bionumpy.bnpdataclass import bnpdataclass

from climate_health.external.models.jax_models.model_spec import Normal, IsDistribution
from climate_health.external.models.jax_models.protoype_annotated_spec import Positive
from climate_health.external.models.jax_models.utii import state_or_param, PydanticTree, get_state_transform

hierarchical = lambda name: state_or_param


@state_or_param
class GlobalParams(PydanticTree):
    alpha: float = 10.
    beta: float = 10.
    sigma: Positive = 1.


@state_or_param
class GlobalSeasonalParams(GlobalParams):
    month_effect: np.ndarray = tuple((0.,)) * 12


@hierarchical('District')
class DistrictParams(PydanticTree):
    alpha: float = 0.
    beta: float = 0.


@bnpdataclass
class Observations:
    x: float
    y: float


@bnpdataclass
class SeasonalObservations(Observations):
    month: int


@state_or_param
class ExpitParams(PydanticTree):
    alpha: float = 0.
    beta: float = 0.
    scale: float = 1.
    location: float = 0.


def expit_transform(params: ExpitParams, x) -> IsDistribution:
    return expit(params.alpha + params.beta * x) * params.scale + params.location
    # return Normal(y_hat, params.sigma)


def linear_regression(params: GlobalParams, given: Observations) -> IsDistribution:
    y_hat = params.alpha + params.beta * given.x
    return Normal(y_hat, params.sigma)


def seasonal_linear_regression(params: GlobalSeasonalParams, given: SeasonalObservations) -> IsDistribution:
    y_hat = params.alpha + params.beta * given.x + params.month_effect[given.month]
    return Normal(y_hat, params.sigma)


def join_global_and_district(global_params: GlobalParams, district_params: DistrictParams) -> GlobalParams:
    return replace(global_params,
                   **{field.name: getattr(global_params, field.name) + getattr(district_params, field.name) for field in
                      dataclasses.fields(district_params)})


def hierarchical_linear_regression(global_params: GlobalParams, district_params: dict[DistrictParams],
                                   given: dict[Observations], regression_model=linear_regression) -> IsDistribution:
    params = {name: join_global_and_district(global_params, district_params[name]) for name in district_params}
    return {name: regression_model(params[name], given[name]) for name in district_params}


@dataclasses.dataclass
class HierarchicalRegression:
    global_params_cls: type
    district_params_cls: type
    observed: dict[str, Any]
    regression_model: Callable = linear_regression

    def prior(self, t_params):
        global_params, district_params = t_params
        return self.global_params_cls().log_prob(global_params) + sum(
            self.district_params_cls().log_prob(district_params[name]) for name in district_params)


def get_hierarchy_logprob_func(global_params_cls, district_params_cls, observed, regression_model=linear_regression,
                               observed_name='y'):
    T_Param, transform, *_ = get_state_transform(global_params_cls)
    T_ParamD, transformD, *_ = get_state_transform(district_params_cls)
    prior = T_Param()
    priorD = T_ParamD()

    def logprob_func(t_params):
        global_params, district_params = t_params
        prior_pdf = prior.log_prob(global_params) + sum(
            priorD.log_prob(district_params[name]) for name in district_params)
        all_params = transform(global_params), {name: transformD(district_params[name]) for name in district_params}
        models = hierarchical_linear_regression(*all_params, observed, regression_model=regression_model)
        observed_probs = [models[name].log_prob(getattr(observed[name], observed_name)).sum() for name in observed]
        obs_pdf = sum(observed_probs)
        return prior_pdf + obs_pdf

    return logprob_func


def get_logprob_func(params_cls, observed, regression_model=linear_regression):
    sampled_y = observed.y
    T_Param, transform, inv_transform = get_state_transform(params_cls)
    prior = T_Param()

    def logprob_func(t_params):
        all_params = transform(t_params)
        return prior.log_prob(t_params) + regression_model(all_params, observed).log_prob(sampled_y).sum()

    return logprob_func


def lagged_regression(params: GlobalParams, given: Sequence[Observations]) -> IsDistribution:
    y_hat = params.alpha + params.beta * given[0].x + params.sigma * given[1].y
    return Normal(y_hat, params.sigma)

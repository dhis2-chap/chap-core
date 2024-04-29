from typing import Sequence

from bionumpy.bnpdataclass import bnpdataclass

from climate_health.external.models.jax_models.model_spec import Normal, IsDistribution
from climate_health.external.models.jax_models.protoype_annotated_spec import Positive
from climate_health.external.models.jax_models.utii import state_or_param, PydanticTree

hierarchical = lambda name: state_or_param

@state_or_param
class GlobalParams(PydanticTree):
    alpha: float = 0.
    beta: float = 0.
    sigma: Positive = 1.


@hiearachical('District')
class DistrictParams:
    alpha: float = 0.
    beta: float = 0.
    sigma: Positive = 1.

#@state_or_param
#class LocalParams:
#    alpha: float = 0

@bnpdataclass
class Observations:
    x: float
    y: float


def linear_regression(params: GlobalParams, given: Observations)->IsDistribution:
    y_hat = params.alpha+params.beta*given.x
    return Normal(y_hat, params.sigma)


def spatial_type(locations: Sequence[str]):
    pass

import pytest

from climate_health.external.models.jax_models.utii import state_or_param, PydanticTree
from jax.tree_util import tree_flatten, tree_unflatten

@state_or_param
class ParamClass(PydanticTree):
    observation_rate: float = 0.01
    year: int = 9

@pytest.fixture
def params():
    return ParamClass()


def test_pydantic_tree(params):
    flat, spec = tree_flatten(params)
    assert flat == [0.01, 9]
    new_params = tree_unflatten(spec, flat)
    assert new_params == params

def test_nested(params):
    d = (params, params)
    flat, spec = tree_flatten(d)
    assert flat == [0.01, 9, 0.01, 9]

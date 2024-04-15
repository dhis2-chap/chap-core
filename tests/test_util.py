import pytest
from climate_health.external.models.jax_models.util import extract_last


def test_extract_last():
    samples={'a': [1, 2, 3],
             'b': {'c': [4, 5, 6]}}
    assert extract_last(samples) == {'a': 3, 'b': {'c': 6}}


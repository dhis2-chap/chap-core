import pytest

from climate_health.external.models.jax_models.model_spec import NutsParams


@pytest.fixture()
def blackjax():
    try:
        import blackjax
    except ImportError:
        pytest.skip("jax is not installed")
    return blackjax

@pytest.fixture()
def jax():
    try:
        import jax
    except ImportError:
        pytest.skip("jax is not installed")
    return jax




@pytest.fixture()
def pm():
    try:
        import pymc3 as pm
    except:
        pytest.skip("pymc3 is not installed")
    return pm


@pytest.fixture()
def fast_params():
    return NutsParams(n_samples=10, n_warmup=10)

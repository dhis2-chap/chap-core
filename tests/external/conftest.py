import pytest

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


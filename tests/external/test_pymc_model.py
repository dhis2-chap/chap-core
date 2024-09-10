import numpy as np


def test_simple_pymc_model(pm):
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sigma=1)
        obs = pm.Normal('obs', mu=mu, sigma=1, observed=np.random.randn(100))
        trace = pm.sample(1000, tune=1000)
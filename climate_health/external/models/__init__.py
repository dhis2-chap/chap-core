models = {}
try:
    from .jax_models.regression_model import RegressionModel
    models['RegressionModel'] = RegressionModel

except ImportError as e:
    print('Could not import RegressionModel')
    print(e)

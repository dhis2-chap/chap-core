models = {}
try:
    from .jax_models.regression_model import RegressionModel, HierarchicalRegressionModel
    from .jax_models.simple_ssm import SSM
    models['RegressionModel'] = RegressionModel
    models['HierarchicalRegressionModel'] = HierarchicalRegressionModel
    models['SSM'] = SSM
except ImportError as e:
    print('Could not import RegressionModel')
    print(e)

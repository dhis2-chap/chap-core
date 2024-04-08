models = {}
try:
    from .jax_models.regression_model import RegressionModel, HierarchicalRegressionModel
    models['RegressionModel'] = RegressionModel
    models['HierarchicalRegressionModel'] = HierarchicalRegressionModel
except ImportError as e:
    print('Could not import RegressionModel')
    print(e)

from chap_core.assessment.evaluator import ComponentBasedEvaluator

import math


def mae_error(truth: float, predictions: list[float]):
    return abs(truth - sum(predictions) / len(predictions))


def mean_across_time(errors):
    return sum(errors) / len(errors)


def mean_across_regions(errors):
    return sum(errors) / len(errors)


def mse_error(truth: float, predictions: list[float]):
    return (truth - sum(predictions) / len(predictions)) ** 2


def sqrt_mean_across_time(errors):
    return math.sqrt(sum(errors) / len(errors))


mae_component_evaluator = ComponentBasedEvaluator("MAE", mae_error, mean_across_time, None)

mae_country_evaluator = ComponentBasedEvaluator("MAE country", mae_error, mean_across_time, mean_across_regions)

absError_timepoint_evaluator = ComponentBasedEvaluator("MAE timpeoint", mae_error, None, None)

rmse_evaluator = ComponentBasedEvaluator("rmse", mse_error, sqrt_mean_across_time, None)


evaluator_suite_options = {
    "onlyLocalMAE": [mae_component_evaluator],
    "localAndGlobalMAE": [mae_component_evaluator, mae_country_evaluator],
    "localMAEandRMSE": [mae_component_evaluator, rmse_evaluator],
    "mix": [mae_component_evaluator, rmse_evaluator, absError_timepoint_evaluator, mae_country_evaluator],
}

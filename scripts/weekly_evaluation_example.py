import dataclasses
from typing import Callable

from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.datatypes import FullData, Samples
from chap_core.predictor.model_registry import registry
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

model = registry.get_model('chap_ewars_weekly')
dataset = DataSet.from_csv('example_data/nicaragua_weekly_data.csv', dataclass=FullData)
if __name__ == '__main__':
    evaluate_model(model,
                   dataset,
                   prediction_length=3,
                   n_test_sets=9,
                   report_filename='nicaragua_example_report.pdf',
                   weather_provider=QuickForecastFetcher)


# def mse(truth, samples):
#     return np.mean((truth - samples) ** 2)
#
#
# class GlontsEvaluation(BaseModel):
#     MSE: float
#     MAPE: float
#
#
#
#
# class Evaluator:
#
#     def __init__(self, func, name, aggregation_func: Callable = sum):
#         self.evaluator_function = func
#
#     @classmethod
#     def from_function(cls, func: Callable, name, aggregation_func: Callable = sum):
#         return cls(func, name, aggregation_func)
#
#     def evaluate_location(self, location, samples, truth):
#         ...
#
#     def evaluate_forecast(self, forecast: dict[str, DataSet[Samples]], truth: DataSet) -> float:
#         for location, samples in forecast.items():
#             truth_data = truth[location]
#             self.evaluate_location(location, samples, truth_data)
#
# @dataclasses.dataclass
# class EvaluatorSuite:
#     evaluators: list[Evaluator]
#
# @evaluator_registry.register_evaluator
# class MySuite:
#
#
# evaluator_registry.register_evaluator('MyEvaluator', EvaluatorSuite([
#     Evaluator.from_function(mse, 'MSE'),
#     Evaluator.from_function(mape, 'MAPE')]
# ))
#
#
# @app.post()
# def evalute_backtest(forecasts: list[BackTestForecast]) -> list[BackTestForecast]:
# ...
#
# func = lambda x: x+2
# obj = EvaluatorBase()
# obj.evaluator_function = func
#
# def cel(truth: bool, samples: list[bool]):
#     return sum(truth == samples) / len(truth)
#
# def make_evauator(metric_func, name, aggregation_func: Optional[Callable] = sum):
#     class Evaluator()
#
#

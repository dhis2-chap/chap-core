from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

from climate_health.dataset import IsSpatioTemporalDataSet
from climate_health.datatypes import HealthData, ResultType, SummaryStatistics


class MultiLocationEvaluator:
    def __init__(self, model_names: List[str], truth: IsSpatioTemporalDataSet):
        self.model_names = model_names
        self.truth = truth
        self.predictions = {model_name: [] for model_name in model_names}

    def add_predictions(self, model_name: str, predictions: IsSpatioTemporalDataSet):
        self.predictions[model_name].append(predictions)
        return self

    def get_results(self) -> dict[str, ResultType]:
        truth_df = self.truth.to_pandas()
        results = {}

        for model_name, predictions in self.predictions.items():
            model_results = []
            for prediction in predictions:
                for location in prediction.locations():

                    pred = prediction.get_location(location).data()
                    pred_time = next(iter(pred.time_period))
                    location_mask = (truth_df['location'] == location)
                    time_mask = truth_df['time_period'] == pred_time.topandas()
                    true = truth_df.loc[location_mask & time_mask]

                    if isinstance(pred, SummaryStatistics):
                        if self.check_data(true.disease_cases, pred.median):
                            mae = mean_absolute_error(true.disease_cases, pred.median)
                            model_results.append([location, str(pred_time.topandas()), mae] + [float(x) for x in [pred.mean, pred.std, pred.median, pred.min, pred.max, pred.quantile_low, pred.quantile_high]])

                    elif isinstance(pred, HealthData):
                        if self.check_data(true.disease_cases, pred.disease_cases):
                            mae = mean_absolute_error(true.disease_cases, pred.disease_cases)
                            model_results.append([location, str(pred_time.topandas()), mae])

            if isinstance(pred, SummaryStatistics):
                results[model_name] = pd.DataFrame(model_results, columns=['location', 'period', 'mae', 'mean', 'std', 'median', 'min', 'max', 'quantile_low', 'quantile_high'])
            elif isinstance(pred, HealthData):
                results[model_name] = pd.DataFrame(model_results, columns=['location', 'period', 'mae'])

        return results

    def check_data(self, true, pred) -> bool:
        if np.isnan(true).any() or np.isnan(pred).any() or len(true) != len(pred):
            return False
        else:
            return True


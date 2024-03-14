from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

from climate_health.dataset import IsSpatioTemporalDataSet
from climate_health.datatypes import HealthData, ResultType


class MultiLocationEvaluator:
    def __init__(self, model_names: List[str], truth: IsSpatioTemporalDataSet[HealthData]):
        self.model_names = model_names
        self.truth = truth
        self.predictions = {model_name: [] for model_name in model_names}

    def add_predictions(self, model_name: str, predictions: IsSpatioTemporalDataSet[HealthData]):
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
                    pred_time = pred.time_period
                    time_str = f"{pred_time.year[0]}-{str(pred_time.month[0]).zfill(2)}" # do smart conversion to str with TimePeriod class?
                    true = truth_df.loc[(truth_df['location'] == location) &            # restrict_time_period method?
                                        (truth_df['time_period'] == time_str)]

                    # check for NaN values
                    if np.isnan(true.disease_cases).any() or np.isnan(pred.disease_cases).any():
                        continue
                    else:
                        mae = mean_absolute_error(true.disease_cases, pred.disease_cases)
                        model_results.append([location, time_str, mae])

            results[model_name] = pd.DataFrame(model_results, columns=['location', 'period', 'mae'])

        return results

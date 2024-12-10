from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from chap_core.data import DataSet
from chap_core.datatypes import HealthData, ResultType, SummaryStatistics


class MultiLocationEvaluator:
    def __init__(self, model_names: List[str], truth: DataSet):
        self.model_names = model_names
        self.truth = truth
        self.predictions = {model_name: [] for model_name in model_names}

    def add_predictions(self, model_name: str, predictions: DataSet):
        self.predictions[model_name].append(predictions)
        return self

    def _mle(self, true, pred):
        return np.log(pred + 1) - np.log(true + 1)

    def get_results(self) -> dict[str, ResultType]:
        # TODO: add split point to dataframe
        # allow multiple observations for each split point
        truth_df = self.truth.to_pandas()
        results = {}

        for model_name, predictions in self.predictions.items():
            model_results = []
            truths = []
            for prediction in predictions:
                for location in prediction.locations():
                    pred = prediction.get_location(location).data()
                    pred_time = next(iter(pred.time_period))
                    location_mask = truth_df["location"] == location
                    time_mask = truth_df["time_period"] == pred_time.topandas()
                    true = truth_df.loc[location_mask & time_mask]

                    true_value = true.disease_cases.values[0]
                    if isinstance(pred, SummaryStatistics):
                        if self.check_data(true.disease_cases, pred.median):
                            mae = mean_absolute_error(true.disease_cases, pred.median)
                            mle = self._mle(true_value + 1, pred.median[0] + 1)
                            new_entry = [
                                location,
                                str(pred_time.topandas()),
                                mae,
                                mle,
                            ] + [
                                float(x)
                                for x in [
                                    pred.mean,
                                    pred.std,
                                    pred.median,
                                    pred.min,
                                    pred.max,
                                    pred.quantile_low,
                                    pred.quantile_high,
                                ]
                            ]
                            truths.append([location, str(pred_time.topandas()), mae, mle] + [float(true_value)] * 7)
                            model_results.append(new_entry)
                            # model_results.append(truth_entry)

                    elif isinstance(pred, HealthData):
                        if self.check_data(true.disease_cases, pred.disease_cases):
                            mae = mean_absolute_error(true.disease_cases, pred.disease_cases)
                            mle = np.log(pred.disease_cases[0] + 1) - np.log(true_value + 1)
                            new_entry = [location, str(pred_time.topandas()), mae, mle]
                            model_results.append(new_entry)

            if isinstance(pred, SummaryStatistics):
                results[model_name] = pd.DataFrame(
                    model_results,
                    columns=[
                        "location",
                        "period",
                        "mae",
                        "mle",
                        "mean",
                        "std",
                        "median",
                        "min",
                        "max",
                        "quantile_low",
                        "quantile_high",
                    ],
                )
                results["truth"] = pd.DataFrame(
                    truths,
                    columns=[
                        "location",
                        "period",
                        "mae",
                        "mle",
                        "mean",
                        "std",
                        "median",
                        "min",
                        "max",
                        "quantile_low",
                        "quantile_high",
                    ],
                )
            elif isinstance(pred, HealthData):
                results[model_name] = pd.DataFrame(model_results, columns=["location", "period", "mae", "mle"])

        return results

    def check_data(self, true, pred) -> bool:
        if np.isnan(true).any() or np.isnan(pred).any() or len(true) != len(pred):
            return False
        else:
            return True

from collections import defaultdict
from typing import Dict

from chap_core.datatypes import ClimateHealthTimeSeries
from chap_core.main import PlaceholderModel, assess_model_on_csv_data, lagged_rows, \
    split_to_train_test_truth_fixed_ahead_lag, make_assessment_report


class MultiLagModelManager:
    """
    Represent and can run multiple models for different lags ahead.
    Also deals with splitting of data.
    """
    def __init__(self, models: Dict[int, PlaceholderModel]):
        self.models = models

    @classmethod
    def from_single_model_class_and_n_lags(cls, single_model: type, n_lags: int):
        models = {i: single_model() for i in range(1, n_lags + 1)}
        return cls(models)

    def fit(self, data, split_fraction: float):
        for lag_ahead in range(1, 10):
            rowbased_data = lagged_rows(data, lag_rows=[3], lag=lag_ahead)
            X_train, Y_train, X_test, Y_test = (
                split_to_train_test_truth_fixed_ahead_lag(rowbased_data, split_fraction))
            self.models[lag_ahead].fit(X_train, Y_train)

    def predict(self, data, split_fraction: float):
        predictions = {}
        for lag_ahead in range(1, 10):
            rowbased_data = lagged_rows(data, lag_rows=[3], lag=lag_ahead)
            X_train, Y_train, X_test, Y_test = (
                split_to_train_test_truth_fixed_ahead_lag(rowbased_data, split_fraction))
            predictions[lag_ahead] = self.models[lag_ahead].predict(X_test)
        return predictions


def assss_model_on_csv_data_with_multi_lag_model(data_file_name: str, split_fraction: float,
                            model: MultiLagModelManager):
    data = ClimateHealthTimeSeries.from_csv(data_file_name)  # real data or simulated data
    data = data.topandas()

    truth_dict = defaultdict(lambda: defaultdict(int))
    for lag_ahead in range(1, 10):
        rowbased_data = lagged_rows(data, lag_rows=[3], lag=lag_ahead)
        X_train, Y_train, X_test, Y_test = (
            split_to_train_test_truth_fixed_ahead_lag(rowbased_data, split_fraction))
        for test_time_offset, (single_X_test, single_Y_test) in enumerate(zip(X_test, Y_test)):
            truth_dict[lag_ahead][test_time_offset] = single_Y_test


    multi_model = MultiLagModelManager.from_single_model_class_and_n_lags(PlaceholderModel, 10)
    multi_model.fit(data, split_fraction)
    prediction_dict = multi_model.predict(data, split_fraction)
    report = make_assessment_report(prediction_dict, truth_dict)
    return report



if __name__ == "__main__":
    file_name = "../example_data/data.csv"
    report = assss_model_on_csv_data_with_multi_lag_model(file_name, 0.5, MultiLagModelManager.from_single_model_class_and_n_lags(PlaceholderModel, 10))
    print(report)
    print(report.text)

from collections import defaultdict
from typing import Dict
from chap_core.datatypes import ClimateHealthTimeSeries
from chap_core.main import PlaceholderModel, assess_model_on_csv_data, lagged_rows, \
    split_to_train_test_truth_fixed_ahead_lag, make_assessment_report


class DataManager:
    def __init__(self, data):
        self.data = data

    def split_to_train_test_truth_fixed_ahead_lag(self, lag, split_fraction):
        data = self.get_lagged_rows([3], lag=lag)
        hardcoded_test_col = 3
        n = data.shape[0]
        train_idx = int((n * split_fraction) // 1)
        X = data.drop(data.columns[hardcoded_test_col], axis=1)
        Y = data.iloc[:, hardcoded_test_col]

        X_train = X.iloc[:train_idx]
        X_test = X.iloc[train_idx:]
        Y_train = Y.iloc[:train_idx]
        Y_test = Y.iloc[train_idx:]

        return X_train, Y_train, X_test, Y_test

    def get_y_test(self, lag, split_fraction):
        return self.split_to_train_test_truth_fixed_ahead_lag(lag, split_fraction)[3]

    def get_x_test(self, lag, split_fraction):
        return self.split_to_train_test_truth_fixed_ahead_lag(lag, split_fraction)[2]

    def get_train(self, lag, split_fraction):
        return self.split_to_train_test_truth_fixed_ahead_lag(lag, split_fraction)[0:2]
    def get_lagged_rows(self, lag_rows, lag=1):
        new_data = self.data.iloc[lag:]
        for lag_row in lag_rows:
            for i in range(lag):
                lagged_col = self.data.iloc[lag - i - 1:-1 - i, lag_row].values
                new_data.insert(len(new_data.columns), f"{self.data.columns[lag_row]}_lag_{i + 1}", lagged_col)
        return new_data

        #
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

    def fit(self, data_manager: DataManager, split_fraction: float):
        for lag_ahead in range(1, 10):
            X_train, Y_train = data_manager.get_train(lag_ahead, split_fraction)
            self.models[lag_ahead].fit(X_train, Y_train)


    def predict(self, data_manager: DataManager, split_fraction: float):
        predictions = {}
        for lag_ahead in range(1, 10):
            x_test = data_manager.get_x_test(lag_ahead, split_fraction)
            predictions[lag_ahead] = self.models[lag_ahead].predict(x_test)
        return predictions


def assss_model_on_csv_data_with_multi_lag_model(data_file_name: str, split_fraction: float,
                            model: MultiLagModelManager):
    data = ClimateHealthTimeSeries.from_csv(data_file_name)  # real data or simulated data
    data = data.topandas()
    data_manager = DataManager(data)

    truth_dict = defaultdict(lambda: defaultdict(int))
    for lag_ahead in range(1, 10):
        y_test = data_manager.get_y_test(lag_ahead, split_fraction)
        for test_time_offset, single_Y_test in enumerate(y_test):
            truth_dict[lag_ahead][test_time_offset] = single_Y_test

    multi_model = MultiLagModelManager.from_single_model_class_and_n_lags(PlaceholderModel, 10)
    multi_model.fit(data_manager, split_fraction)
    prediction_dict = multi_model.predict(data_manager, split_fraction)
    report = make_assessment_report(prediction_dict, truth_dict)
    return report



if __name__ == "__main__":
    file_name = "../example_data/data.csv"
    report = assss_model_on_csv_data_with_multi_lag_model(file_name, 0.5, MultiLagModelManager.from_single_model_class_and_n_lags(PlaceholderModel, 10))
    print(report)
    print(report.text)

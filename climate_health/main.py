"""
Contains function for running full analysis (e.g from data to prediction assesment report)
"""


import pandas as pd
from collections import defaultdict
from climate_health.datatypes import ClimateHealthTimeSeries


def assess_model_on_csv_data(data_file_name: str, split_fraction: float,
                             model: 'PlaceholderModel') -> 'AssessmentReport':
    """
    Wraps together all necessary steps for reading data from file, assessing a model on different lags ahead
    and generating a report with results.
    """
    data = ClimateHealthTimeSeries.from_csv(data_file_name)
    data = data.topandas()

    prediction_dict = defaultdict(lambda: defaultdict(int))
    truth_dict = defaultdict(lambda: defaultdict(int))
    for lag_ahead in range(1, 10):
        rowbased_data = lagged_rows(data, lag_rows=[3], lag=lag_ahead)
        x_train, y_train, x_test, y_test = (
            split_to_train_test_truth_fixed_ahead_lag(rowbased_data, split_fraction))
        model.fit(x_train, y_train)
        for test_time_offset, (single_X_test, single_Y_test) in enumerate(zip(x_test, y_test)):
            prediction_dict[lag_ahead][test_time_offset] = model.predict(single_X_test)
            truth_dict[lag_ahead][test_time_offset] = single_Y_test
    report = make_assessment_report(prediction_dict, truth_dict)
    return report


def lagged_rows(data, lag_rows, lag=1):
    new_data = data.iloc[lag:]
    for lag_row in lag_rows:
        for i in range(lag):
            lagged_col = data.iloc[lag - i - 1:-1 - i, lag_row].values
            new_data.insert(len(new_data.columns), f"{data.columns[lag_row]}_lag_{i + 1}", lagged_col)
    return new_data


def split_to_train_test_truth_fixed_ahead_lag(data, split_fraction):
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


class PlaceholderModel:

    def __init__(self):
        return

    def fit(self, x_train, y_train):
        return self

    def predict(self, single_x_test):
        """
        Takes a single test data point and returns prediction
        """
        return 0


class AssessmentReport:
    def __init__(self, text):
        self.text = text
        return


def make_assessment_report(prediction_dict, truth_dict) -> AssessmentReport:
    return AssessmentReport("Good job!")

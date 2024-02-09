import pandas as pd
from collections import defaultdict
from climate_health.datatypes import ClimateHealthTimeSeries

def lagged_rows(data, lag_rows, lag=1):
    new_data = data.iloc[lag:]
    for lag_row in lag_rows:
        for i in range(lag):
            lagged_col = data.iloc[lag - i - 1:-1 - i, lag_row].values
            new_data.insert(len(new_data.columns), f"{data.columns[lag_row]}_lag_{i + 1}", lagged_col)
    return new_data


# %%
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


# %%
class PlaceholderModel:

    def __init__(self):
        return

    def fit(self, X_train, Y_train):
        return self

    def predict(self, single_X_test):
        return 0


# %%
class AssessmentReport:
    def __init__(self, text):
        self.text = text
        return


# %%
def make_assessment_report(prediction_dict, truth_dict) -> AssessmentReport:
    return AssessmentReport("Good job!")


def main_with_outer_lagged(fn: str, split_fraction: float,
                           model: PlaceholderModel) -> AssessmentReport:
    data = ClimateHealthTimeSeries.from_csv(fn)  # real data or simulated data
    data = data.topandas()

    prediction_dict = defaultdict(lambda: defaultdict(int))
    truth_dict = defaultdict(lambda: defaultdict(int))
    for lag_ahead in range(1, 2):
        rowbased_data = lagged_rows(data, lag_rows=[3], lag=lag_ahead)
        X_train, Y_train, X_test, Y_test = split_to_train_test_truth_fixed_ahead_lag(rowbased_data, split_fraction)
        model.fit(X_train, Y_train)
        for test_time_offset, (single_X_test, single_Y_test) in enumerate(zip(X_test, Y_test)):
            prediction_dict[lag_ahead][test_time_offset] = model.predict(single_X_test)
            truth_dict[lag_ahead][test_time_offset] = single_Y_test
    report = make_assessment_report(prediction_dict, truth_dict)
    return report


if __name__ == "__main__":
    report = main_with_outer_lagged("../example_data/data.csv", 0.5, PlaceholderModel())
    print(report.text)

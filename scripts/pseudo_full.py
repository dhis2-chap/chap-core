from chap_core.datatypes import ClimateHealthTimeSeries

class ClimateHealthTimeSeriesWithLags:
    ...


class ClimateHealthTimeSeriesWithLagsSingleLine:
    ...


class ChPredictionMethodWithSpecificFeatureConfig:
    ...

class AssessmentReport:
    pass

class ChTrustedPredictionMultiModelManager:
    pass


def make_data_rowbased_specific_version(data : ClimateHealthTimeSeries, lag_ahead: int) -> ClimateHealthTimeSeriesWithLags:
    pass

def split_to_train_test_truth_fixed_ahead_lag(rowbased_data: ClimateHealthTimeSeriesWithLags, now_timepoint : int, lag_ahead) -> tuple[ClimateHealthTimeSeriesWithLags, ClimateHealthTimeSeriesWithLags, tuple[int]]:
    pass

def make_data_rowbased_specific_feature_config(data: ClimateHealthTimeSeries, lag_ahead: int) -> ClimateHealthTimeSeriesWithLags:
    pass


def assess(prediction: int, truth: int) -> float:
    pass


def make_assessment_report(prediction_dict: dict[int,dict[int,float]], truth_dict: dict[int,dict[int,int]]) -> AssessmentReport:
    pass



from collections import defaultdict


def main_with_outer_lagged(fn: str, now_timepoint: int, method: ChPredictionMethodWithSpecificFeatureConfig) -> AssessmentReport:
    data = ClimateHealthTimeSeries.from_csv(fn)  # real data or simulated data
    prediction_dict = defaultdict(dict)
    truth_dict = {}
    for lag_ahead in range(1, 10):
        rowbased_data = make_data_rowbased_specific_feature_config(data, lag_ahead)
        x_train, y_train, x_tests, y_tests = split_to_train_test_truth_fixed_ahead_lag(rowbased_data, now_timepoint)
        model = method.fit(x_train, y_train)
        # predict one time point individually to avoid leaking of truth
        for test_time_offset, (test, truth) in enumerate(zip(x_tests, y_tests)):
            prediction_dict[lag_ahead][test_time_offset] = model.predict(test)
            truth_dict[lag_ahead][test_time_offset] = truth
    report = make_assessment_report(prediction_dict, truth_dict)
    return report





class ChTrustedPredictionMultiModelManager:
    def load_data(self, data):
        pass

    def configure(self, lag_ahead):
        pass

    def prepare_lagged_versions(self):
        pass

    def prepare_train_test_truth(self):
        pass

    def fit_managed_models(self, data):
        pass

    def run(self, data, lag_ahead):
        self._configure(lag_ahead)
        self._load_data(data)
        self._prepare_lagged_versions()
        self._prepare_train_test_truth()
        self._fit_managed_models(data)


def main_with_multi_time_ahead_predictor(fn : str, now_timepoint: int, manager:ChTrustedPredictionMultiModelManager[ChPredictionMethodWithSpecificFeatureConfig]):
    data = ClimateHealthTimeSeries.from_csv(fn)
    prediction_dict = defaultdict(dict)
    truth_dict = {}
    lag_ahead = range(1, 10)

    #rowbased_data = make_data_rowbased_specific_feature_config(data, lag_ahead)
    #train, tests, truths = split_to_train_test_truth_fixed_ahead_lag(rowbased_data, now_timepoint, lag_ahead)

    predictions, truths = manager.run(data, lag_ahead)

    for test_time_offset, (test, truth) in enumerate(zip(tests, truths)):
        prediction_dict[lag_ahead][test_time_offset] = model.predict(test)
        truth_dict[lag_ahead][test_time_offset] = truth
    report = make_assessment_report(prediction_dict, truth_dict)
    return report

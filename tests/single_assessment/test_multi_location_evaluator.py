from chap_core.assessment.multi_location_evaluator import MultiLocationEvaluator
from chap_core.datatypes import SummaryStatistics
from chap_core.time_period import Month, PeriodRange
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import pytest


@pytest.fixture
def r_model_predictions():
    time_period = PeriodRange.from_time_periods(Month(2012, 8), Month(2012, 8))
    T = len(time_period)
    d = {
        "oslo": SummaryStatistics(
            time_period,
            mean=[0.5] * T,
            std=[0.5] * T,
            median=[0.5] * T,
            min=[0.5] * T,
            max=[0.5] * T,
            quantile_low=[0.5] * T,
            quantile_high=[0.5] * T,
        ),
        "bergen": SummaryStatistics(
            time_period,
            mean=[0.5] * T,
            std=[0.5] * T,
            median=[0.5] * T,
            min=[0.5] * T,
            max=[0.5] * T,
            quantile_low=[0.5] * T,
            quantile_high=[0.5] * T,
        ),
    }
    return DataSet(d)


# @pytest.mark.xfail
def test_multi_location_evaluator(full_data, good_predictions, bad_predictions):
    evaluator = MultiLocationEvaluator(
        model_names=["bad_model", "good_model"], truth=full_data
    )
    evaluator.add_predictions("good_model", good_predictions)
    evaluator.add_predictions("bad_model", bad_predictions)
    results = evaluator.get_results()

    for i in range(len(results["good_model"]["mae"])):
        assert results["good_model"]["mae"][i] < results["bad_model"]["mae"][i]


def test_multi_location_evaluator_r_model(
    full_data, good_predictions, r_model_predictions
):
    evaluator = MultiLocationEvaluator(
        model_names=["good_model", "r_model"], truth=full_data
    )
    evaluator.add_predictions("good_model", good_predictions)
    evaluator.add_predictions("r_model", r_model_predictions)
    results = evaluator.get_results()

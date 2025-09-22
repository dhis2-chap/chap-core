import logging

import pytest

logger = logging.getLogger(__name__)

from chap_core.assessment.data_representation_transforming import (
    convert_to_multi_location_forecast,
    convert_to_multi_location_timeseries,
    MAEonMeanPredictions,
)


# @pytest.mark.xfail(reason="Failing. Under development?")
def test_external_evaluation(backtest):
    dataset = backtest.dataset
    f2 = list(convert_to_multi_location_forecast(backtest.forecasts).values())[0]
    t2 = convert_to_multi_location_timeseries(dataset.observations)
    t2 = t2.filter_by_time_periods(f2.time_periods())
    mae = MAEonMeanPredictions().evaluate(t2, f2)
    print(mae)
    logger.warning(
        "We need to convert this to a format that chap can work on. Or create db entries for aggregated metrics and type ans specify different types of results."
    )

from chap_core.assessment.data_representation_transforming import (
    convert_to_multi_location_forecast,
    convert_to_multi_location_timeseries,
    MAEonMeanPredictions,
)


def test_external_evaluation(backtest):
    dataset = backtest.dataset
    f2 = list(convert_to_multi_location_forecast(backtest.forecasts).values())[0]
    t2 = convert_to_multi_location_timeseries(dataset.observations)
    t2 = t2.filter_by_time_periods(f2.time_periods())
    mae = MAEonMeanPredictions().evaluate(t2, f2)
    assert set(mae.timeseries_dict) == set(f2.timeseries)

"""Metrics computed from forecast cells, and forecasts with missing samples."""

import numpy as np
import pandas as pd
import pytest

from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.forecast_cells import ForecastCells, MissingSamplesError
from chap_core.assessment.metrics import available_metrics, calculate_metrics
from chap_core.assessment.metrics.crps_norm import CRPSNormMetric
from chap_core.datatypes import SamplesWithTruth
from chap_core.rest_api.data_models import BacktestCreate
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange


def test_from_flat_gives_one_row_per_cell_with_its_observation(flat_observations, flat_forecasts_multiple_samples):
    observations = flat_observations[flat_observations.time_period == "2023-W01"]

    cells = ForecastCells.from_flat(observations, flat_forecasts_multiple_samples)

    assert cells.keys.to_dict("list") == {
        "location": ["loc1", "loc1", "loc2", "loc2"],
        "time_period": ["2023-W01", "2023-W02", "2023-W01", "2023-W02"],
        "horizon_distance": [1, 2, 1, 2],
    }
    np.testing.assert_array_equal(cells.samples[0], [8.0, 10.0, 15.0])
    np.testing.assert_array_equal(cells.observed, [11.0, np.nan, 19.0, np.nan])


def test_from_flat_rejects_forecasts_with_fewer_samples(flat_observations, flat_forecasts_multiple_samples):
    with pytest.raises(MissingSamplesError):
        ForecastCells.from_flat(flat_observations, flat_forecasts_multiple_samples.iloc[1:])


def test_crps_norm_only_uses_observations_that_have_a_forecast(flat_observations, flat_forecasts_multiple_samples):
    outside = pd.DataFrame({"location": ["loc1"], "time_period": ["2020-W01"], "disease_cases": [1000.0]})
    with_outside = pd.concat([flat_observations, outside], ignore_index=True)

    metric = CRPSNormMetric()

    pd.testing.assert_frame_equal(
        metric.get_global_metric(with_outside, flat_forecasts_multiple_samples),
        metric.get_global_metric(flat_observations, flat_forecasts_multiple_samples),
    )


def test_metrics_from_cells_match_metrics_from_flat_tables(backtest, tmp_path):
    path = tmp_path / "evaluation.nc"
    Evaluation.from_backtest(backtest).to_file(path)
    evaluation = Evaluation.from_file(path)
    flat = evaluation.to_flat()

    from_cells = calculate_metrics(evaluation, list(available_metrics))

    for metric_id, metric_cls in available_metrics.items():
        expected = float(metric_cls().get_global_metric(flat.observations, flat.forecasts)["metric"].iloc[0])
        assert from_cells[metric_id] == pytest.approx(expected, nan_ok=True), metric_id


def _samples_with_truth(samples: np.ndarray) -> DataSet:
    periods = PeriodRange.from_strings(["2024-01", "2024-02"])
    return DataSet(
        {"loc1": SamplesWithTruth(time_period=periods, samples=samples, disease_cases=np.array([1.0, 2.0]))}  # type: ignore[call-arg]
    )


def test_from_samples_with_truth_rejects_forecasts_with_some_samples_missing(mocker):
    samples = np.ones((2, 3))
    samples[0, 1] = np.nan
    configured_model = mocker.MagicMock(id=1)

    with pytest.raises(MissingSamplesError):
        Evaluation.from_samples_with_truth(
            [_samples_with_truth(samples)],
            last_train_period=PeriodRange.from_strings(["2023-12"])[0],
            configured_model=configured_model,
            info=BacktestCreate(name="test", dataset_id=0, model_id="model"),
        )

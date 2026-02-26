from pathlib import Path

import altair
import pandas as pd
import pytest

from chap_core.assessment.backtest_plots import (
    get_backtest_plots_registry,
    get_backtest_plot,
    list_backtest_plots,
    create_plot_from_backtest,
    create_plot_from_evaluation,
    BacktestPlotBase,
)
from chap_core.assessment.backtest_plots.evaluation_plot import EvaluationPlot, _infer_split_periods
from chap_core.assessment.backtest_plots.horizon_location_grid import HorizonLocationGridPlot
from chap_core.assessment.backtest_plots.metrics_dashboard import MetricsDashboard
from chap_core.assessment.backtest_plots.sample_bias_plot import SampleBiasPlot
from chap_core.assessment.evaluation import Evaluation
from chap_core.cli_endpoints.utils import plot_backtest
from chap_core.database.tables import BackTest


@pytest.fixture(scope="module")
def default_transformer():
    altair.data_transformers.enable("default")
    yield


def test_backtest_plot_registry():
    """Test that all plots are registered in the registry."""
    registry = get_backtest_plots_registry()

    # Check that expected plots are registered
    assert "metrics_dashboard" in registry
    assert "ratio_of_samples_above_truth" in registry
    assert "evaluation_plot" in registry

    # Check that all registered plots are subclasses of BacktestPlotBase
    for plot_id, plot_cls in registry.items():
        assert issubclass(plot_cls, BacktestPlotBase)


def test_get_backtest_plot():
    """Test getting a specific plot by ID."""
    plot_cls = get_backtest_plot("metrics_dashboard")
    assert plot_cls is MetricsDashboard

    plot_cls = get_backtest_plot("ratio_of_samples_above_truth")
    assert plot_cls is SampleBiasPlot

    plot_cls = get_backtest_plot("evaluation_plot")
    assert plot_cls is EvaluationPlot

    # Test non-existent plot
    assert get_backtest_plot("non_existent") is None


def test_list_backtest_plots():
    """Test listing all backtest plots."""
    plots = list_backtest_plots()

    # Check that it returns a list of dicts with expected keys
    assert len(plots) >= 3
    for plot in plots:
        assert "id" in plot
        assert "name" in plot
        assert "description" in plot
        assert "needs_historical" in plot


def test_evaluation_plot_directly(flat_observations, flat_forecasts, default_transformer):
    """Test the evaluation plot with flat data."""
    plot = EvaluationPlot()
    chart = plot.plot(pd.DataFrame(flat_observations), pd.DataFrame(flat_forecasts))
    assert chart is not None


def test_sample_bias_plot_directly(flat_observations, flat_forecasts, default_transformer):
    """Test the sample bias plot with flat data."""
    plot = SampleBiasPlot()
    chart = plot.plot(pd.DataFrame(flat_observations), pd.DataFrame(flat_forecasts))
    assert chart is not None


def test_metrics_dashboard_directly(flat_observations, flat_forecasts, default_transformer):
    """Test the metrics dashboard with flat data."""
    plot = MetricsDashboard()
    chart = plot.plot(pd.DataFrame(flat_observations), pd.DataFrame(flat_forecasts))
    assert chart is not None


def test_horizon_location_grid_directly(flat_observations, flat_forecasts_multiple_samples, default_transformer):
    """Test the horizon location grid plot with multiple-sample forecasts."""
    plot = HorizonLocationGridPlot()
    chart = plot.plot(pd.DataFrame(flat_observations), pd.DataFrame(flat_forecasts_multiple_samples))
    assert chart is not None


def test_infer_split_periods_monthly_format():
    """Test that _infer_split_periods produces date strings, not repr strings like 'Month(2022-1)'."""
    df = pd.DataFrame(
        {
            "location": ["loc1", "loc1"],
            "time_period": ["2022-01", "2022-02"],
            "horizon_distance": [1, 2],
            "q_50": [10.0, 12.0],
        }
    )
    result = _infer_split_periods(df)
    for split_period in result["split_period"]:
        assert "Month(" not in split_period, f"Got repr string: {split_period}"
        assert split_period.startswith("20"), f"Unexpected format: {split_period}"


def test_evaluation_plot_monthly_data(default_transformer):
    """Test that evaluation plot works with monthly time periods."""
    forecasts = pd.DataFrame(
        {
            "location": ["loc1"] * 4 + ["loc2"] * 4,
            "time_period": ["2022-01", "2022-02"] * 2 + ["2022-01", "2022-02"] * 2,
            "horizon_distance": [1, 2, 1, 2, 1, 2, 1, 2],
            "sample": [0, 0, 1, 1, 0, 0, 1, 1],
            "forecast": [10.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0],
        }
    )
    observations = pd.DataFrame(
        {
            "location": ["loc1", "loc1", "loc2", "loc2"],
            "time_period": ["2022-01", "2022-02", "2022-01", "2022-02"],
            "disease_cases": [11.0, 13.0, 19.0, 21.0],
        }
    )
    plot = EvaluationPlot()
    chart = plot.plot(observations, forecasts)
    assert chart is not None


@pytest.mark.parametrize("plot_id", list(get_backtest_plots_registry().keys()))
def test_all_registered_plots_from_backtest(plot_id: str, simulated_backtest: BackTest, default_transformer):
    """Test that all registered plots can be successfully generated from a BackTest."""
    chart = create_plot_from_backtest(plot_id, simulated_backtest)
    assert chart is not None


@pytest.mark.parametrize("plot_id", list(get_backtest_plots_registry().keys()))
def test_all_registered_plots_from_evaluation(plot_id: str, simulated_backtest: BackTest, default_transformer):
    """Test that all registered plots can be successfully generated from an Evaluation."""
    evaluation = Evaluation.from_backtest(simulated_backtest)
    chart = create_plot_from_evaluation(plot_id, evaluation)
    assert chart is not None


def test_plot_backtest_cli(backtest: BackTest, tmp_path: Path, default_transformer):
    """Test the CLI plot_backtest function."""
    evaluation = Evaluation.from_backtest(backtest)
    input_file = tmp_path / "evaluation.nc"
    evaluation.to_file(input_file)

    output_file = tmp_path / "plot.html"
    plot_backtest(input_file, output_file, plot_type="metrics_dashboard")

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_generate_pdf_report(backtest: BackTest, tmp_path: Path):
    from chap_core.cli_endpoints.utils import generate_pdf_report

    evaluation = Evaluation.from_backtest(backtest)
    input_file = tmp_path / "evaluation.nc"
    evaluation.to_file(input_file)

    output_file = tmp_path / "report.pdf"
    generate_pdf_report(input_file, output_file)

    assert output_file.exists()
    assert output_file.stat().st_size > 0

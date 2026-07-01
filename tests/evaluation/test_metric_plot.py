import altair as alt
import pytest
from pathlib import Path

from chap_core.assessment.metric_plots import (
    MetricPlotBase,
    get_metric_plots_registry,
    list_metric_plots,
    metric_plot,
)
from chap_core.assessment.metric_plots.horizon_mean import MetricByHorizonV2Mean
from chap_core.assessment.metric_plots.horizon_sum import MetricByHorizonV2Sum
from chap_core.assessment.metric_plots.metric_map import MetricMapV2
from chap_core.assessment.metric_plots.regional_distribution import RegionalMetricDistributionPlot
from chap_core.assessment.metric_plots.time_period_mean import MetricByTimePeriodV2Mean
from chap_core.assessment.metric_plots.time_period_sum import MetricByTimePeriodV2Sum
from chap_core.assessment.metrics.crps import CRPSMetric


@pytest.fixture
def metric_data(flat_observations, flat_forecasts):
    return CRPSMetric().get_detailed_metric(flat_observations, flat_forecasts)


def test_metric_plot_registry():
    registry = get_metric_plots_registry()
    assert "metric_by_horizon_mean" in registry
    assert "metric_map" in registry
    assert "regional_metric_distribution" in registry
    for plot_cls in registry.values():
        assert issubclass(plot_cls, MetricPlotBase)


def test_list_metric_plots():
    plots = list_metric_plots()
    assert isinstance(plots, list)
    for entry in plots:
        assert "id" in entry
        assert "name" in entry
        assert "description" in entry


def test_metric_plot_decorator_sets_attributes(monkeypatch):
    import chap_core.assessment.metric_plots as _mp

    isolated_registry: dict = {}
    monkeypatch.setattr(_mp, "_metric_plots_registry", isolated_registry)

    @metric_plot(plot_id="test_decorator_plot", name="Test Plot", description="For testing only")
    class _TestPlot(MetricPlotBase):
        def plot_from_df(self, title: str = "") -> alt.Chart:
            return alt.Chart()

    assert _TestPlot.id == "test_decorator_plot"
    assert _TestPlot.name == "Test Plot"
    assert _TestPlot.description == "For testing only"
    assert "test_decorator_plot" in get_metric_plots_registry()


def test_metric_plot_decorator_rejects_non_subclass():
    with pytest.raises(TypeError):

        @metric_plot(plot_id="bad_plot", name="Bad")
        class _NotAPlot:
            pass


@pytest.mark.parametrize(
    "plot_cls",
    get_metric_plots_registry().values(),
)
def test_registered_plot_produces_chart(plot_cls, metric_data, dummy_geojson):
    chart = plot_cls(metric_data, dummy_geojson).plot_from_df()
    assert chart is not None


@pytest.mark.parametrize(
    "plot_cls",
    [MetricByHorizonV2Sum, MetricByTimePeriodV2Sum, MetricByHorizonV2Mean, MetricByTimePeriodV2Mean],
)
def test_metric_plots_pre_aggregate_in_pandas(plot_cls, metric_data, dummy_geojson):
    # These plots aggregate in pandas before charting, so the compiled Vega spec must
    # not carry an in-spec aggregate transform (which would embed the raw, un-aggregated
    # rows for the browser to reduce).
    spec = plot_cls(metric_data, dummy_geojson).plot_from_df().to_dict(format="vega")
    for data in spec.get("data") or []:
        for transform in data.get("transform") or []:
            ops = transform.get("ops") or []
            assert not (transform.get("type") == "aggregate" and ({"sum", "mean"} & set(ops)))


def test_regional_metric_distribution_empty_data(metric_data):
    empty = metric_data.iloc[0:0]
    chart = RegionalMetricDistributionPlot(empty).plot_from_df()
    spec = chart.to_dict(format="vega")
    assert any(isinstance(m, dict) and m.get("type") == "text" for m in (spec.get("marks") or []))

# --- dependencies ---
import altair as alt
import pandas as pd
import textwrap
import yaml

from chap_core.database.tables import BackTest
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics import available_metrics

from chap_core.plotting.evaluation_plot import (
    MetricByHorizonV2Mean,
    MetricByHorizonV2Sum,
    MetricByHorizonAndLocationMean,
    MetricByTimePeriodV2Mean,
    MetricByTimePeriodV2Sum,
    MetricByTimePeriodAndLocationV2Mean,
    MetricMapV2,
)

alt.data_transformers.enable("vegafusion")

"""
    YAML structure example:

    name: TestPlot
    title: Overskrift som vises øverst
    configuration:
      - row:
          - plot:
              type: metric
              metric: detailed_crps
              plot_type: metric_by_horizon_mean
          - text:
              value: "Mean Detailed CRPS by horizon."
      - row:
          - plot:
              type: metric
              metric: samples_above_truth
              plot_type: metric_by_time_mean
          - text:
              value: "Samples above truth by time period."
"""

METRIC_PLOT_TYPES = {
    "metric_by_horizon_mean": MetricByHorizonV2Mean,
    "metric_by_horizon_sum": MetricByHorizonV2Sum,
    "metric_by_horizon_location_mean": MetricByHorizonAndLocationMean,
    "metric_by_time_mean": MetricByTimePeriodV2Mean,
    "metric_by_time_sum": MetricByTimePeriodV2Sum,
    "metric_by_time_location_mean": MetricByTimePeriodAndLocationV2Mean,
    "metric_map": MetricMapV2,
}


def title_chart(text: str, width: int = 600, font_size: int = 24, pad: int = 10):
    return (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
        .mark_text(
            text=text,
            fontSize=font_size,
            fontWeight="bold",
            align="center",
            baseline="top",
        )
        .properties(width=width, height=font_size + pad)
    )


def text_chart(text, line_length=80, font_size=12, align="left", pad_bottom=50):
    lines = textwrap.wrap(str(text), width=line_length)
    df = pd.DataFrame({"line": lines, "y": range(len(lines))})
    line_spacing = font_size + 2
    total_height = len(lines) * line_spacing + pad_bottom
    return (
        alt.Chart(df)
        .mark_text(align=align, baseline="top", fontSize=font_size)
        .encode(text="line", y=alt.Y("y:O", axis=None))
        .properties(height=total_height)
    )


def _concat_h(charts):
    charts = [c for c in charts if c is not None]
    if not charts:
        return None
    out = charts[0]
    for c in charts[1:]:
        out = out | c
    return out


def _concat_v(charts):
    charts = [c for c in charts if c is not None]
    if not charts:
        return None
    out = charts[0]
    for c in charts[1:]:
        out = out & c
    return out


def _build_plot_component(comp: dict, context: dict):
    comp_type = comp.get("type") or comp.get("kind")

    if comp_type == "metric":
        metric_name = comp.get("metric")
        plot_type = comp.get("plot_type")

        if not metric_name:
            return text_chart("Metric name missing in YAML component.", line_length=60)

        metric_class = available_metrics.get(metric_name)
        if metric_class is None:
            return text_chart(
                f"Metric '{metric_name}' not found.\nAvailable: {', '.join(sorted(available_metrics.keys()))}",
                line_length=60,
            )

        flat_obs = context["flat_observations"]
        flat_fcst = context["flat_forecasts"]

        metric = metric_class()

        metric_df = metric.compute(flat_obs, flat_fcst)

        plot_class = METRIC_PLOT_TYPES.get(plot_type)
        if plot_class is None:
            return text_chart(
                f"Unknown plot_type '{plot_type}' for metric '{metric_name}'.\n"
                f"Available plot types: {', '.join(sorted(METRIC_PLOT_TYPES.keys()))}",
                line_length=60,
            )

        required_cols = []

        missing = [c for c in required_cols if c not in metric_df.columns]
        if missing:
            return text_chart(
                f"Plot type '{plot_type}' requires columns {missing}, "
                f"but metric '{metric_name}' returned columns {list(metric_df.columns)}.",
                line_length=80,
            )

        plotter = plot_class(metric_df)
        return plotter.plot()

    return text_chart(f"Unknown plot kind: {comp_type}", line_length=60)


def build_from_yaml(yaml_str: str, backtest: BackTest, **context):
    configuration = yaml.safe_load(yaml_str)

    title = configuration.get("title")
    rows_spec = configuration.get("configuration", [])

    if backtest is None:
        raise ValueError("build_from_yaml requires a real BackTest object.")

    evaluation = Evaluation.from_backtest(backtest)
    flat_data = evaluation.to_flat()
    flat_obs = flat_data.observations
    flat_fcst = flat_data.forecasts

    context = {
        **context,
        "flat_observations": flat_obs,
        "flat_forecasts": flat_fcst,
    }

    vrows = []
    if title:
        vrows.append(title_chart(title))

    for row in rows_spec:
        components = row.get("row", [])
        hcharts = []
        for comp in components:
            if "plot" in comp:
                chart = _build_plot_component(comp["plot"], context)
                hcharts.append(chart)
            elif "text" in comp:
                txt = comp["text"]
                if isinstance(txt, dict):
                    val = txt.get("value", "")
                else:
                    val = str(txt)
                hcharts.append(text_chart(val, line_length=60))
            else:
                hcharts.append(text_chart("Unknown component", line_length=60))
        vrows.append(_concat_h(hcharts))

    chart = _concat_v([ch for ch in vrows if ch is not None])
    return chart

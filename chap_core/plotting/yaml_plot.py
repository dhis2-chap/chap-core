# --- dependencies ---
import altair as alt
import pandas as pd
import textwrap
import yaml

from chap_core.database.tables import BackTest
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics import available_metrics

from chap_core.plotting.backtest_plot import EvaluationBackTestPlot
from chap_core.plotting.evaluation_plot import (
    MetricByHorizonV2Mean,
    MetricByHorizonV2Sum,
    MetricByHorizonAndLocationMean,
    MetricByTimePeriodV2Mean,
    MetricByTimePeriodV2Sum,
    MetricByTimePeriodAndLocationV2Mean,
    MetricMapV2,
)

# If you want vegafusion (as in chap-core), you can keep this.
# If you want pure inline data (e.g. for generic Vega viewers), switch to default/inline.
alt.data_transformers.enable("vegafusion")
# alt.data_transformers.enable("default")   # or "inline" if you prefer


# -------------------------------------------------------------------
# Metric plot types -> MetricPlotV2 subclasses
# -------------------------------------------------------------------
METRIC_PLOT_TYPES = {
    "metric_by_horizon_mean": MetricByHorizonV2Mean,
    "metric_by_horizon_sum": MetricByHorizonV2Sum,
    "metric_by_horizon_location_mean": MetricByHorizonAndLocationMean,
    "metric_by_time_mean": MetricByTimePeriodV2Mean,
    "metric_by_time_sum": MetricByTimePeriodV2Sum,
    "metric_by_time_location_mean": MetricByTimePeriodAndLocationV2Mean,
    "metric_map": MetricMapV2,
}


# -------------------------------------------------------------------
# Simple helpers: title, text, concatenation
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Plot component builder
# -------------------------------------------------------------------
def _build_plot_component(backtest: BackTest, comp: dict, context: dict):
    """
    Build a single plot component from a YAML 'plot' block.

    Supported:
      type: metric
        metric: <key in available_metrics>
        plot_type: <key in METRIC_PLOT_TYPES>
      type: evaluation
        (no extra fields)
    """
    comp_type = comp.get("type") or comp.get("kind")

    # --- METRIC COMPONENTS ---
    if comp_type == "metric":
        metric_name = comp.get("metric")
        plot_type = comp.get("plot_type", "metric_by_horizon_mean")

        if not metric_name:
            return text_chart("Metric name missing in YAML component.", line_length=60)

        metric_cls = available_metrics.get(metric_name)
        if metric_cls is None:
            return text_chart(
                f"Metric '{metric_name}' not found.\n"
                f"Available: {', '.join(sorted(available_metrics.keys()))}",
                line_length=60,
            )

        flat_obs = context["flat_observations"]
        flat_fcst = context["flat_forecasts"]

        metric = metric_cls()

        # CHAP-style: evaluation_plot uses .compute(...)
        # (If your MetricBase only has get_metric, swap to that.)
        metric_df = metric.compute(flat_obs, flat_fcst)

        plot_class = METRIC_PLOT_TYPES.get(plot_type)
        if plot_class is None:
            return text_chart(
                f"Unknown plot_type '{plot_type}' for metric '{metric_name}'.\n"
                f"Available plot types: {', '.join(sorted(METRIC_PLOT_TYPES.keys()))}",
                line_length=60,
            )

        # Optional: basic compatibility checks to avoid cryptic KeyErrors
        required_cols = []
        if plot_type.startswith("metric_by_time"):
            required_cols.append("time_period")
        if "horizon" in plot_type:
            required_cols.append("horizon_distance")

        missing = [c for c in required_cols if c not in metric_df.columns]
        if missing:
            return text_chart(
                f"Plot type '{plot_type}' requires columns {missing}, "
                f"but metric '{metric_name}' returned columns {list(metric_df.columns)}.",
                line_length=80,
            )

        plotter = plot_class(metric_df)
        return plotter.plot()

    # --- EVALUATION PLOT: forecast vs truth over time ---
    if comp_type == "evaluation":
        # Use the dedicated backtest forecast-vs-truth plot
        return EvaluationBackTestPlot.from_backtest(backtest).plot()

    # Fallback
    return text_chart(f"Unknown plot kind: {comp_type}", line_length=60)


# -------------------------------------------------------------------
# MAIN: build dashboard from YAML + BackTest
# -------------------------------------------------------------------
def build_from_yaml(yaml_str: str, backtest: BackTest, **context):
    """
    YAML structure example:

    name: TestPlot
    title: Overskrift som vises øverst
    configuration:
      - row:
          - plot:
              type: evaluation
          - text:
              value: "Backtest forecast vs truth."
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
    # Parse YAML into python dict
    configuration = yaml.safe_load(yaml_str)

    # Retrieve data
    title = configuration.get("title")
    rows_spec = configuration.get("configuration", [])

    if backtest is None:
        raise ValueError("build_from_yaml requires a real BackTest object.")

    # Use CHAP Evaluation abstraction to get flat forecasts/observations
    evaluation = Evaluation.from_backtest(backtest)
    flat_data = evaluation.to_flat()
    flat_obs = flat_data.observations
    flat_fcst = flat_data.forecasts

    # Extend context with flat data
    context = {
        **context,
        "flat_observations": flat_obs,
        "flat_forecasts": flat_fcst,
    }

    vrows = []
    if title:
        vrows.append(title_chart(title))

    for r in rows_spec:
        components = r.get("row", [])
        hcharts = []
        for c in components:
            if "plot" in c:
                chart = _build_plot_component(backtest, c["plot"], context)
                hcharts.append(chart)
            elif "text" in c:
                txt = c["text"]
                if isinstance(txt, dict):
                    val = txt.get("value", "")
                else:
                    val = str(txt)
                hcharts.append(text_chart(val, line_length=60))
            else:
                hcharts.append(text_chart("Unknown component", line_length=60))
        vrows.append(_concat_h(hcharts))

    # Compose final chart
    chart = _concat_v([ch for ch in vrows if ch is not None])
    return chart


# -------------------------------------------------------------------
# Example usage (you must supply a real BackTest from CHAP)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # TODO: replace this with how you actually obtain a BackTest
    # e.g. from DB, API, or a serialized JSON
    #
    # Example placeholder:
    #   from chap_core.database.session import get_session
    #   from chap_core.crud.backtests import get_backtest
    #
    #   session = get_session()
    #   backtest = get_backtest(session, backtest_id="some-uuid")
    #
    # For now, we just mark it as a variable you must define:
    backtest: BackTest = ...  # <-- supply a real BackTest here

    yaml_cfg = """
    name: TestPlot
    title: Overskrift som vises øverst
    configuration:
      - row:
          - plot:
              type: evaluation
          - text:
              value: "Forecast vs observed disease cases."
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

    chart = build_from_yaml(yaml_cfg, backtest)
    chart.save("yaml_plot.json")
    print("Saved yaml_plot.json")

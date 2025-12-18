# --- dependencies you already use ---
import altair as alt
import pandas as pd
import numpy as np
import textwrap
import yaml
from collections.abc import Mapping
from datetime import datetime
from chap_core.assessment.metrics import available_metrics
from chap_core.assessment.evaluation import Evaluation
from chap_core.plotting.evaluation_plot import (
    MetricByHorizonV2Mean,
    MetricByHorizonV2Sum,
    MetricByHorizonAndLocationMean,
    MetricByTimePeriodV2Mean,
    MetricByTimePeriodV2Sum,
    MetricByTimePeriodAndLocationV2Mean,
    MetricMapV2,
)

METRIC_PLOT_TYPES = {
    "metric_by_horizon_mean": MetricByHorizonV2Mean,
    "metric_by_horizon_sum": MetricByHorizonV2Sum,
    "metric_by_horizon_location_mean": MetricByHorizonAndLocationMean,
    "metric_by_time_mean": MetricByTimePeriodV2Mean,
    "metric_by_time_sum": MetricByTimePeriodV2Sum,
    "metric_by_time_location_mean": MetricByTimePeriodAndLocationV2Mean,
    "metric_map": MetricMapV2,
}


# ---------- title/text helpers (from your file) ----------
def title_chart(text: str, width: int = 600, font_size: int = 24, pad: int = 10):
    return (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
        .mark_text(text=text, fontSize=font_size, fontWeight="bold", align="center", baseline="top")
        .properties(width=width, height=font_size + pad)
    )


def text_chart(text, line_length=80, font_size=12, align="left", pad_bottom=50):
    lines = textwrap.wrap(text, width=line_length)
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
    if not charts:
        return None
    out = charts[0]
    for c in charts[1:]:
        out = out | c
    return out


def _concat_v(charts):
    if not charts:
        return None
    out = charts[0]
    for c in charts[1:]:
        out = out & c
    return out


# ---------- plot component builder ----------
def _build_plot_component(backtest, comp: dict, context: dict):
    type = comp.get("type")
    if type == "metric":
        metric_name = comp.get("metric")
        plot_type = comp.get("plot_type", "bar")  # -> Default

        if not metric_name:
            return text_chart("Metric name missing in YAML component.", line_length=60)

        metric_cls = available_metrics.get(metric_name)
        if metric_cls is None:
            return text_chart(
                f"Metric '{metric_name}' not found. Available: {', '.join(sorted(available_metrics.keys()))}",
                line_length=60,
            )

        obs = context.get("flat_observations")
        fcst = context.get("flat_forecasts")

        metric = metric_cls()
        metric_df = metric.get_metric(obs, fcst)

        print("=== METRIC:", metric_name, "PLOT_TYPE:", plot_type, "===")
        print(metric_df.head())
        print(metric_df.columns)
        print("rows:", len(metric_df))

        plot_class = METRIC_PLOT_TYPES.get(plot_type)
        if plot_class is None:
            return text_chart(
                f"Unknown plot_type '{plot_type}' for metric '{metric_name}'.",
                line_length=60,
            )
        plotter = plot_class(metric_df)
        return plotter.plot()

    return text_chart(f"Unknown plot kind: {type}", line_length=60)


# ---------- MAIN: build_from_yaml ----------
def build_from_yaml(yaml_str: str, backtest, **context):
    """
    name: TestPlot
    title: Overskrift som vises øverst
    configuration:
        - row:
            - plot:
                type: metric
                metric: crps_per_location
                plot_type: metric_by_horizon_mean
            - text:
                value: "Litt tekst underveis."
        - row:
            - plot:
                type: metric
                metric: above_truth
            - text:
                value: "Ratio of Samples Above Truth (per time & horizon)."
    """
    # Parse YAML into python dict
    configuration = yaml.safe_load(yaml_str)

    # Retrieve data
    title = configuration.get("title")
    rows_spec = configuration.get("configuration", [])

    # Prepare flat data
    if "flat_observations" in context and "flat_forecasts" in context:
        flat_obs = context["flat_observations"]
        flat_fcst = context["flat_forecasts"]
    else:
        if backtest is None:
            raise ValueError(
                "build_from_yaml: either provide a real BackTest or pass "
                "'flat_observations' and 'flat_forecasts' in context."
            )
        evaluation = Evaluation.from_backtest(backtest)
        flat_data = evaluation.to_flat()
        flat_obs = flat_data.observations
        flat_fcst = flat_data.forecasts

    # overwrite / extend context with normalized keys
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


class ForecastStub:
    def __init__(self, period, org_unit, last_seen_period, quantiles):
        """
        quantiles can be:
        - Mapping: {0.1: v, 0.25: v, 0.5: v, 0.75: v, 0.9: v}
        - float/int/np.floating: single value -> replicated across quantiles
        - sequence aligned to [0.1, 0.25, 0.5, 0.75, 0.9]
        """
        self.period = period
        self.org_unit = org_unit
        self.last_seen_period = last_seen_period

        qkeys = [0.1, 0.25, 0.5, 0.75, 0.9]
        if isinstance(quantiles, Mapping):
            qd = {float(k): (None if quantiles[k] is None else float(quantiles[k])) for k in quantiles}
        elif isinstance(quantiles, (list, tuple, np.ndarray)):
            qd = {
                qkeys[i]: (None if quantiles[i] is None else float(quantiles[i]))
                for i in range(min(len(qkeys), len(quantiles)))
            }
        elif isinstance(quantiles, (int, float, np.floating)) or quantiles is None:
            val = None if quantiles is None else float(quantiles)
            qd = {q: val for q in qkeys}
        else:
            qd = {q: None for q in qkeys}
        for q in qkeys:
            qd.setdefault(q, None)
        self._qs = qd

    def get_quantiles(self, qs):
        return [self._qs.get(float(q)) for q in qs]


class DatasetStub:
    def __init__(self, observations):
        self.observations = observations


class BackTestStub:
    def __init__(self, forecasts, dataset):
        self.forecasts = forecasts
        self.dataset = dataset


class ObservationStub:
    def __init__(self, period, org_unit, feature_name, value):
        self.period = period  # e.g. '2023-01-02'
        self.org_unit = org_unit  # e.g. 'loc1'
        self.feature_name = feature_name  # 'disease_cases'
        self.value = value  # float/int


# ---------- main ----------
if __name__ == "__main__":
    DATA_DIR = "../../../../metrics/Assessment_example_chap_compatible/example_data"
    forecasts_path = DATA_DIR + "/forecasts.csv"
    obs_path = DATA_DIR + "/observations.csv"

    def week_to_date(s):
        dt = datetime.strptime(f"{s}-1", "%G-W%V-%u")
        return dt.strftime("%Y-%m-%d")

    # load
    fc = pd.read_csv(forecasts_path)
    obs = pd.read_csv(obs_path)

    # just convert time_period to real dates; keep everything else as-is
    fc_std = fc.copy()
    fc_std["time_period"] = fc_std["time_period"].astype(str).map(week_to_date)

    obs_std = obs.copy()
    obs_std["time_period"] = obs_std["time_period"].astype(str).map(week_to_date)

    # YAML
    yaml_cfg = """
    name: TestPlot
    title: Overskrift som vises øverst
    configuration:
        - row:
            - plot:
                type: metric
                metric: detailed_crps
                plot_type: metric_by_horizon_mean
            - text:
                value: "CRPS by horizon mean."
        - row:
            - plot: 
                type: metric
                metric: samples_above_truth
                plot_type: metric_by_horizon_mean
            - text:
                value: "Ratio of Samples Above Truth (per location & horizon)."
        - row: 
            - plot: 
                type: metric
                metric: detailed_rmse
                plot_type: metric_by_time_location_mean
            - text: 
                value: "Detailed RMSE by time period and location mean."
        - row: 
            - plot: 
                type: metric
                metric: peak_value_diff
                plot_type: metric_by_time_location_mean
            - text: 
                value: "Peak Value Difference by time period and location mean."
    """

    # 🔹 Pass flat_observations and flat_forecasts directly
    chart = build_from_yaml(
        yaml_cfg,
        backtest=None,  # we don't use Evaluation here
        flat_observations=obs_std,
        flat_forecasts=fc_std,
    )

    chart.save("backtest_dashboard_from_yaml.json")
    print("✅ Saved backtest_dashboard_from_yaml.json")

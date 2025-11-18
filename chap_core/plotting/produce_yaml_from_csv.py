# --- dependencies you already use ---
import altair as alt
import pandas as pd
import numpy as np
import textwrap
import yaml
from collections.abc import Mapping
import numpy as np
from datetime import datetime, timedelta
from chap_core.plotting.backtest_plot import EvaluationBackTestPlot

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

# ---------- small utilities ----------
def _ensure_df_date_str(df, col):
    """Ensure date-like strings for Altair."""
    out = df.copy()
    out[col] = out[col].astype(str)
    return out

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

# ---------- metric implementations ----------
def _metric_crps_per_location_chart(backtest, observations_df: pd.DataFrame, samples_df: pd.DataFrame | None = None):
    """
    Approx CRPS per location: when forecasts are deterministic (or single sample),
    CRPS equals absolute error. We compute |median_forecast - truth| per (location, time),
    then average by location and (optionally) by horizon (via last_seen_period gap).
    """
    # Build a small dataframe of median forecasts from backtest.forecasts
    rows = []
    for f in backtest.forecasts:
        q50 = f.get_quantiles([0.5])[0]
        rows.append({
            "location": f.org_unit,
            "time_period": str(f.period),
            "last_seen_period": str(f.last_seen_period),
            "forecast_median": q50,
        })
    fmed = pd.DataFrame(rows)

    obs = observations_df.rename(columns={"time_period": "time_period", "location": "location", "disease_cases": "disease_cases"}).copy()
    obs["time_period"] = obs["time_period"].astype(str)

    df = fmed.merge(obs, on=["location", "time_period"], how="inner")
    if df.empty:
        # Friendly placeholder if merge failed
        return text_chart("Ingen overlapp mellom forecasts og observasjoner for å beregne CRPS.", line_length=60)

    df["abs_err"] = (df["forecast_median"] - df["disease_cases"]).abs()

    # Estimate horizon via time gap (string dates sorted lexicographically OK for ISO date strings)
    # If you already track horizon elsewhere, replace this with that column.
    # Here we just keep last_seen_period to facet/group by it if present.
    agg = df.groupby(["location"], as_index=False)["abs_err"].mean().rename(columns={"abs_err": "crps_approx"})

    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("location:N", title="Location"),
            y=alt.Y("crps_approx:Q", title="CRPS (approx)"),
            tooltip=["location", alt.Tooltip("crps_approx:Q", format=".3f")],
        )
        .properties(title="CRPS per location (approx)")
    )
    return chart

# ---------- evaluation plot ----------
def _evaluation_chart(backtest):
    # Uses your EvaluationBackTestPlot class
    plot = EvaluationBackTestPlot.from_backtest(backtest)
    return plot.plot()

# ---------- dataset 'map' fallback ----------
def _dataset_chart(data: pd.DataFrame, kind: str):
    """
    Simple visual fallback: heatmap or line plot of disease_cases by location/time.
    """
    df = data.rename(columns={"time_period": "time_period", "location": "location", "disease_cases": "disease_cases"}).copy()
    df = _ensure_df_date_str(df, "time_period")

    if kind == "disease_cases_map":
        # Heatmap (time x location) as a lightweight stand-in for a map
        return (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt.X("time_period:T", title="Time"),
                y=alt.Y("location:N", title="Location"),
                color=alt.Color("disease_cases:Q", title="Cases"),
                tooltip=["location", "time_period", "disease_cases"]
            )
            .properties(title="Disease cases (heatmap)")
        )
    else:
        # Line chart per location
        return (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("time_period:T"),
                y=alt.Y("disease_cases:Q"),
                color="location:N",
                tooltip=["location", "time_period", "disease_cases"]
            )
            .properties(title="Disease cases over time")
        )

# ---------- plot component builder ----------
def _build_plot_component(backtest, comp: dict, context: dict):
    kind = comp.get("kind")
    if kind == "metric":
        metric_name = comp.get("metric")
        plot_type = comp.get("plot_type")
        # We only implement crps_per_location for now; extend here for more metrics/plot_types
        if metric_name == "crps_per_location":
            obs = context.get("main")
            samples = context.get("samples")  # optional
            return _metric_crps_per_location_chart(backtest, obs, samples)
        return text_chart(f"Metric '{metric_name}' not implemented yet.", line_length=60)

    if kind == "evaluation":
        return _evaluation_chart(backtest)

    if kind == "dataset":
        data_key = comp.get("data_key", "main")
        plot_type = comp.get("plot_type", "line")
        df = context.get(data_key)
        if df is None:
            return text_chart(f"Dataset key '{data_key}' missing from context.", line_length=60)
        return _dataset_chart(df, plot_type)

    return text_chart(f"Unknown plot kind: {kind}", line_length=60)

# ---------- MAIN: build_from_yaml ----------
def build_from_yaml(yaml_str: str, backtest, **context):
    """
    Minimal YAML-driven dashboard builder.

    Supported schema:
    name: <str>
    title: <str>
    layout:
      - row:
        - plot: { kind: metric|evaluation|dataset, ... }
        - text: { value: <str> }
      - row:
        - plot: ...
        - text: ...
    """
    cfg = yaml.safe_load(yaml_str)

    # Optional top title
    title = cfg.get("title")
    rows_spec = cfg.get("layout", [])

    vrows = []
    if title:
        vrows.append(title_chart(title))

    for r in rows_spec:
        # each row is a dict with key 'row' -> list of components
        comps = r.get("row", [])
        hcharts = []
        for c in comps:
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
            qd = {qkeys[i]: (None if quantiles[i] is None else float(quantiles[i])) for i in range(min(len(qkeys), len(quantiles)))}
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
        self.period = period              # e.g. '2023-01-02'
        self.org_unit = org_unit          # e.g. 'loc1'
        self.feature_name = feature_name  # 'disease_cases'
        self.value = value                # float/int

# ---------- main ----------
if __name__ == "__main__":
    # your CSV directory
    DATA_DIR = "../../../../metrics/Assessment_example_chap_compatible/example_data"
    forecasts_path = DATA_DIR + "/forecasts.csv"
    obs_path = DATA_DIR + "/observations.csv"

    # ISO week -> date
    def week_to_date(s):
        dt = datetime.strptime(f"{s}-1", "%G-W%V-%u")
        return dt.strftime("%Y-%m-%d")

    def minus_weeks(iso_date_str, k):
        dt = datetime.fromisoformat(iso_date_str)
        return (dt - timedelta(weeks=int(k))).strftime("%Y-%m-%d")

    # load
    fc = pd.read_csv(forecasts_path)
    obs = pd.read_csv(obs_path)

    # convert time
    fc["period_date"] = fc["time_period"].astype(str).map(week_to_date)
    fc["last_seen_date"] = [minus_weeks(p, h) for p, h in zip(fc["period_date"], fc["horizon_distance"])]
    obs_std = obs.copy()
    obs_std["time_period"] = obs_std["time_period"].astype(str).map(week_to_date)

    obs_objects = [
        ObservationStub(period=row["time_period"], org_unit=row["location"],
                        feature_name="disease_cases", value=row["disease_cases"])
        for _, row in obs_std.iterrows()
    ]    

    # quantiles per (location, period, last_seen)
    def qvals(series):
        arr = series.to_numpy()
        return {
            0.10: float(np.quantile(arr, 0.10)),
            0.25: float(np.quantile(arr, 0.25)),
            0.50: float(np.quantile(arr, 0.50)),
            0.75: float(np.quantile(arr, 0.75)),
            0.90: float(np.quantile(arr, 0.90)),
        }

    qs = (
        fc.groupby(["location", "period_date", "last_seen_date"])["forecast"]
          .apply(qvals)
          .reset_index(name="qdict")
    )

    forecasts = [
        ForecastStub(row["period_date"], row["location"], row["last_seen_date"], row["qdict"])
        for _, row in qs.iterrows()
    ]
    bt = BackTestStub(forecasts, DatasetStub(obs_objects))

    # YAML
    yaml_cfg = """
    name: TestPlot
    title: Overskrift som vises øverst
    layout:
      - row:
        - plot:
            kind: metric
            metric: crps_per_location
            plot_type: metric_by_horizon_mean
            title: "CRPS per horisont"
        - text:
            value: "Litt tekst underveis."
      - row:
        - plot:
            kind: evaluation
            title: "Truth vs forecast"
      - row:
        - plot:
            kind: dataset
            plot_type: disease_cases_map
            data_key: main
    """

    chart = build_from_yaml(yaml_cfg, bt, main=obs_std)
    chart.save("backtest_dashboard_from_yaml.json")
    print("✅ Saved backtest_dashboard_from_yaml.json")
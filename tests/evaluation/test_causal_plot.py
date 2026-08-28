import json
from pathlib import Path

import altair as alt
import pandas as pd
import pytest

from chap_core.assessment.causal_plot import plot_counterfactual, plot_counterfactual_overlayed
from chap_core.assessment.evaluation import Evaluation
from chap_core.database.dataset_tables import DataSet, Observation
from chap_core.database.tables import Backtest, BacktestForecast

_EXAMPLE_DATA = Path(__file__).parent.parent.parent / "example_data"


def _load_vietnam_geojson(locations: list[str]) -> str:
    raw = json.loads((_EXAMPLE_DATA / "vietnam_monthly.geojson").read_text())
    raw["features"] = [f for f in raw["features"] if f["id"] in locations]
    return json.dumps(raw)


def _make_vietnam_evaluation(df: pd.DataFrame, periods: list[str], geojson: str, scale: float) -> Evaluation:
    obs = [
        Observation(
            feature_name="disease_cases",
            id=i,
            dataset_id=1,
            period=row["time_period"],
            org_unit=row["location"],
            value=float(row["disease_cases"]),
        )
        for i, (_, row) in enumerate(df.iterrows())
    ]
    dataset = DataSet(id=1, name="vietnam", type="test", geojson=geojson, covariates=[], observations=obs, created=None)
    forecasts = [
        BacktestForecast(
            id=i,
            backtest_id=1,
            period=row["time_period"],
            org_unit=row["location"],
            last_train_period=periods[0],
            last_seen_period=periods[0],
            values=[float(row["disease_cases"]) * scale + j for j in range(3)],
        )
        for i, (_, row) in enumerate(df.iterrows())
    ]
    backtest = Backtest(
        id=1,
        dataset_id=1,
        dataset=dataset,
        model_id="test",
        model_db_id=1,
        name="vietnam_test",
        created=None,
        aggregate_metrics={},
        forecasts=forecasts,
        metrics=[],
    )
    return Evaluation.from_backtest(backtest)


@pytest.fixture(scope="module")
def vietnam_evaluation_pair():
    df = pd.read_csv(_EXAMPLE_DATA / "vietnam_monthly.csv")
    locations = sorted(df["location"].unique())[:2]
    df = df[df["location"].isin(locations)]
    periods = sorted(df["time_period"].unique())[-6:]
    df = df[df["time_period"].isin(periods)].reset_index(drop=True)
    geojson = _load_vietnam_geojson(locations)
    return (
        _make_vietnam_evaluation(df, periods, geojson, scale=1.0),
        _make_vietnam_evaluation(df, periods, geojson, scale=0.7),
    )


def test_plot_counterfactual_returns_chart(vietnam_evaluation_pair):
    eval_orig, eval_cf = vietnam_evaluation_pair
    assert plot_counterfactual(eval_orig, eval_cf, ["rainfall"]) is not None


def test_plot_counterfactual_is_vconcat(vietnam_evaluation_pair):
    eval_orig, eval_cf = vietnam_evaluation_pair
    assert isinstance(plot_counterfactual(eval_orig, eval_cf), alt.VConcatChart)


def test_plot_counterfactual_saves_html(vietnam_evaluation_pair, tmp_path):
    eval_orig, eval_cf = vietnam_evaluation_pair
    out = tmp_path / "causal.html"
    plot_counterfactual(eval_orig, eval_cf, ["rainfall"]).save(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_counterfactual_title_with_columns(vietnam_evaluation_pair):
    eval_orig, eval_cf = vietnam_evaluation_pair
    chart = plot_counterfactual(eval_orig, eval_cf, ["rainfall", "temperature"])
    assert "rainfall, temperature" in chart.title


def test_plot_counterfactual_title_without_columns(vietnam_evaluation_pair):
    eval_orig, eval_cf = vietnam_evaluation_pair
    chart = plot_counterfactual(eval_orig, eval_cf, None)
    title = chart.title
    assert "(" not in title and ")" not in title


def test_plot_counterfactual_custom_dataset_labels(vietnam_evaluation_pair):
    eval_orig, eval_cf = vietnam_evaluation_pair
    chart = plot_counterfactual(eval_orig, eval_cf, None, original_label="Baseline", counterfactual_label="Scenario A")
    subplot_titles = {panel["title"] for row in chart.to_dict()["vconcat"] for panel in row["hconcat"]}
    assert {"Baseline", "Scenario A"} <= subplot_titles
    assert chart.title == "Causal Analysis: Baseline vs Scenario A"


def _make_vietnam_evaluation_with_history(
    df: pd.DataFrame, periods: list[str], geojson: str, scale: float, split_index: int
) -> Evaluation:
    """Like _make_vietnam_evaluation but the first `split_index` periods are historical context
    (no forecasts) and the rest are the forecast horizon."""
    historical_periods = set(periods[:split_index])
    forecast_df = df[~df["time_period"].isin(historical_periods)].reset_index(drop=True)
    last_train_period = periods[split_index - 1]

    obs = [
        Observation(
            feature_name="disease_cases",
            id=i,
            dataset_id=1,
            period=row["time_period"],
            org_unit=row["location"],
            value=float(row["disease_cases"]),
        )
        for i, (_, row) in enumerate(forecast_df.iterrows())
    ]
    dataset = DataSet(id=1, name="vietnam", type="test", geojson=geojson, covariates=[], observations=obs, created=None)
    forecasts = [
        BacktestForecast(
            id=i,
            backtest_id=1,
            period=row["time_period"],
            org_unit=row["location"],
            last_train_period=last_train_period,
            last_seen_period=last_train_period,
            values=[float(row["disease_cases"]) * scale + j for j in range(3)],
        )
        for i, (_, row) in enumerate(forecast_df.iterrows())
    ]
    backtest = Backtest(
        id=1,
        dataset_id=1,
        dataset=dataset,
        model_id="test",
        model_db_id=1,
        name="vietnam_test",
        created=None,
        aggregate_metrics={},
        forecasts=forecasts,
        metrics=[],
    )
    historical_observations = [
        Observation(
            feature_name="disease_cases",
            id=1000 + i,
            dataset_id=1,
            period=row["time_period"],
            org_unit=row["location"],
            value=float(row["disease_cases"]),
        )
        for i, (_, row) in enumerate(df[df["time_period"].isin(historical_periods)].iterrows())
    ]
    return Evaluation(backtest, historical_observations=historical_observations, historical_context_periods=split_index)


@pytest.fixture(scope="module")
def vietnam_evaluation_pair_with_history():
    df = pd.read_csv(_EXAMPLE_DATA / "vietnam_monthly.csv")
    locations = sorted(df["location"].unique())[:2]
    df = df[df["location"].isin(locations)]
    periods = sorted(df["time_period"].unique())[-6:]
    df = df[df["time_period"].isin(periods)].reset_index(drop=True)
    geojson = _load_vietnam_geojson(locations)
    return (
        _make_vietnam_evaluation_with_history(df, periods, geojson, scale=1.0, split_index=3),
        _make_vietnam_evaluation_with_history(df, periods, geojson, scale=0.7, split_index=3),
    )


def _layer_frames(chart: alt.VConcatChart) -> list[pd.DataFrame]:
    spec = chart.to_dict()
    datasets = spec.get("datasets", {})
    frames = []
    for panel in spec["vconcat"]:
        for layer in panel["layer"]:
            data = layer["data"]
            values = data["values"] if "values" in data else datasets[data["name"]]
            frames.append(pd.DataFrame(values))
    return frames


def test_plot_overlayed_returns_chart(vietnam_evaluation_pair):
    eval_orig, eval_cf = vietnam_evaluation_pair
    assert plot_counterfactual_overlayed(eval_orig, eval_cf, ["rainfall"]) is not None


def test_plot_overlayed_is_vconcat(vietnam_evaluation_pair):
    eval_orig, eval_cf = vietnam_evaluation_pair
    assert isinstance(plot_counterfactual_overlayed(eval_orig, eval_cf), alt.VConcatChart)


def test_plot_overlayed_saves_html(vietnam_evaluation_pair, tmp_path):
    eval_orig, eval_cf = vietnam_evaluation_pair
    out = tmp_path / "causal_overlayed.html"
    plot_counterfactual_overlayed(eval_orig, eval_cf, ["rainfall"]).save(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_overlayed_custom_title(vietnam_evaluation_pair):
    eval_orig, eval_cf = vietnam_evaluation_pair
    chart = plot_counterfactual_overlayed(eval_orig, eval_cf, ["rainfall"], title="My Title")
    assert chart.title == "My Title"


def test_plot_overlayed_default_title_includes_columns(vietnam_evaluation_pair):
    eval_orig, eval_cf = vietnam_evaluation_pair
    chart = plot_counterfactual_overlayed(eval_orig, eval_cf, ["rainfall", "temperature"])
    assert "rainfall, temperature" in chart.title


def test_plot_overlayed_axis_labels(vietnam_evaluation_pair):
    eval_orig, eval_cf = vietnam_evaluation_pair
    spec = plot_counterfactual_overlayed(eval_orig, eval_cf, ["rainfall"], x_label="Month", y_label="Cases").to_dict()
    encodings = [layer["encoding"] for panel in spec["vconcat"] for layer in panel["layer"]]
    assert any(e["x"].get("title") == "Month" for e in encodings)
    assert any(e["y"].get("title") == "Cases" for e in encodings)


def test_plot_overlayed_custom_dataset_labels(vietnam_evaluation_pair):
    eval_orig, eval_cf = vietnam_evaluation_pair
    chart = plot_counterfactual_overlayed(
        eval_orig, eval_cf, ["rainfall"], original_label="Baseline", counterfactual_label="Scenario A"
    )
    spec = chart.to_dict()
    color_domains = [
        layer["encoding"]["color"]["scale"]["domain"] for panel in spec["vconcat"] for layer in panel["layer"]
    ]
    assert color_domains and all(d[:2] == ["Baseline", "Scenario A"] for d in color_domains)
    assert chart.title == "Causal Analysis: Baseline vs Scenario A (rainfall)"


def test_plot_overlayed_observed_truncated_at_split(vietnam_evaluation_pair_with_history):
    eval_orig, eval_cf = vietnam_evaluation_pair_with_history
    chart = plot_counterfactual_overlayed(eval_orig, eval_cf, ["rainfall"])

    observed_periods = set()
    forecast_periods = set()
    for frame in _layer_frames(chart):
        if "disease_cases" in frame.columns:
            observed_periods.update(frame["time_period"])
        if "q_50" in frame.columns:
            forecast_periods.update(frame["time_period"])

    assert observed_periods
    assert forecast_periods
    assert max(observed_periods) < min(forecast_periods)

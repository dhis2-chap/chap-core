import json
from pathlib import Path

import altair as alt
import pandas as pd
import pytest

from chap_core.assessment.causal_plot import plot_counterfactual
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


def test_plot_counterfactual_returns_chart(vietnam_evaluation_pair, default_transformer):
    eval_orig, eval_cf = vietnam_evaluation_pair
    assert plot_counterfactual(eval_orig, eval_cf, ["rainfall"]) is not None


def test_plot_counterfactual_is_vconcat(vietnam_evaluation_pair, default_transformer):
    eval_orig, eval_cf = vietnam_evaluation_pair
    assert isinstance(plot_counterfactual(eval_orig, eval_cf), alt.VConcatChart)


def test_plot_counterfactual_saves_html(vietnam_evaluation_pair, default_transformer, tmp_path):
    eval_orig, eval_cf = vietnam_evaluation_pair
    out = tmp_path / "causal.html"
    plot_counterfactual(eval_orig, eval_cf, ["rainfall"]).save(str(out))
    assert out.exists() and out.stat().st_size > 0

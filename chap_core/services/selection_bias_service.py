"""Selection bias of picking the best model on a backtest specification, estimated from its own split points.

Picking the model with the best score over the splits flatters that model: part of its lead
is luck, and it is gone on new data. The bias is estimated with a nested scheme inside the
splits. Every window of consecutive splits [a, b) plays validation data and picks the model
with the best comparison with the baseline over it. The pick is then compared, on every later
split t >= b, with the final model f (the one picked on all n splits):

    gap = value of f on split t - value of the window's pick on split t

The gap is zero whenever the window picks f, so a model that wins every window has no
selection bias, however its score moves over time. Both values are on the same split, so a
change that hits every model alike (drift) cancels out. The final model was picked with split
t included, so a window that picks another model shows how much of f's lead on t was luck.

The gap shrinks like the noise of a k-split mean, with k = b - a the window size, and goes to
zero for long windows. It is fitted by least squares through the origin,

    gap = inverse_sqrt_k / sqrt(k),

weighted by the number of forecasts scored in the test split. The estimated bias of the final
model is the fit at k = n, and its estimated value on new data is its validation value minus
that bias. d = t - b, the number of splits between the window and the test split, is kept for
a check: with drift cancelled, the gap should not change with d.

Scores pool org units and, by default, every horizon distance, so the final model and its
validation value are the ones the leaderboard ranks on. Only forecasts every candidate and the
baseline made are scored, so all models are compared on the same forecasts.

With several horizon distances, the forecasts of neighbouring split points share target
periods: the last split of a window and the first splits after it forecast some of the same
months. A test split that shares a month with the window would not be independent of the pick,
so the test splits start a gap of (largest - smallest horizon distance) splits after the window,
and d counts from there. The gap assumes consecutive split points (stride 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field
from sqlmodel import col, select

from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
from chap_core.assessment.metrics import get_comparison_description, get_metric
from chap_core.assessment.metrics.base import AggregationOp
from chap_core.database.base_tables import DBModel
from chap_core.database.database import SessionWrapper
from chap_core.database.settings import SettingsManager
from chap_core.database.tables import (
    Backtest,
    BacktestSpecification,
    SpecificationBaseline,
    SpecificationBaselineStatus,
    is_not_archived_model_backtest,
    is_not_baseline_backtest,
)
from chap_core.services.prediction_monitoring_service import origin_periods
from chap_core.time_period import TimePeriod

if TYPE_CHECKING:
    import pandas as pd
    from sqlmodel import Session

    from chap_core.assessment.metrics.base import Metric

KEYS = ["location", "time_period", "horizon_distance"]


class SpecificationNotFoundError(Exception):
    """Raised when the backtest specification does not exist."""


class SelectionBiasError(Exception):
    """Raised when the analysis cannot run: unknown metric or baseline, too few models or splits."""


class SelectionBiasPoint(DBModel):
    """One point of a scatter plot of the gap against the window size or the distance."""

    x: int = Field(description="Window size k (splits) or distance d (splits after the window and the gap).")
    value: float = Field(
        description="Mean gap of the pairs at this x, weighted by forecasts scored. On the distance plot it is "
        "adjusted to the full window size, using the fitted coefficient."
    )
    fitted: float = Field(description="The fitted line at this x.")
    n_pairs: int = Field(description="Number of (validation window, test split) pairs at this x.")


class SelectionBiasCoefficients(DBModel):
    """Coefficient of gap = inverse_sqrt_k / sqrt(k)."""

    inverse_sqrt_k: float = Field(description="Selection effect: coefficient of 1 / sqrt(window size).")


class SelectionBiasCandidate(DBModel):
    """A candidate model and its comparison with the baseline over all split points."""

    backtest_id: int = Field(description="Id of the candidate's backtest.")
    configured_model_id: int = Field(description="Id of the candidate's configured model.")
    model_name: str = Field(description="Name of the candidate's configured model.")
    validation_value: float = Field(description="Comparison with the baseline over all split points.")


class SelectionBias(DBModel):
    """Estimated selection bias of the best model on a specification, with the data for its two plots."""

    metric_id: str = Field(description="Id of the metric that was compared with the baseline.")
    baseline_model: str = Field(description="Name of the baseline model the values are compared with.")
    comparison_description: str = Field(description="One-line description of what the comparison values mean.")
    horizon_distances: list[int] = Field(description="Horizon distances whose forecasts were scored.")
    gap_splits: int = Field(
        description="Splits left out between each validation window and its first test split, so the two "
        "share no target period: the largest minus the smallest horizon distance scored."
    )
    split_periods: list[str] = Field(description="Split points used, in order: the periods each split predicted from.")
    candidates: list[SelectionBiasCandidate] = Field(description="Every candidate model, best first.")
    selected_backtest_id: int = Field(description="Backtest of the model picked on all split points.")
    validation_value: float = Field(
        description="The picked model's comparison with the baseline over all split points."
    )
    estimated_bias: float = Field(
        description="Fitted gap at the full window size: how much of the picked model's lead is expected to be luck. "
        "Zero when every validation window picks the same model."
    )
    estimated_value: float = Field(description="Estimated comparison on new data: validationValue - estimatedBias.")
    mean_gap: float = Field(
        description="Mean gap over all pairs, weighted by forecasts scored. Higher than the estimated bias, because "
        "most pairs have short windows."
    )
    n_pairs: int = Field(description="Number of (validation window, test split) pairs.")
    n_windows: int = Field(description="Number of validation windows.")
    n_windows_other_pick: int = Field(
        description="Number of validation windows that pick another model than the final one."
    )
    coefficients: SelectionBiasCoefficients
    window_size_points: list[SelectionBiasPoint] = Field(description="Gap against window size k.")
    distance_points: list[SelectionBiasPoint] = Field(
        description="Gap against distance d, at the full window size. A check: it should be flat."
    )


@dataclass
class SplitScores:
    """Per-split sums of a mean-aggregated metric, on the forecasts every model made."""

    model_sums: np.ndarray  # (splits, models)
    baseline_sums: np.ndarray  # (splits,)
    counts: np.ndarray  # (splits,), forecasts scored


def _comparisons(metric: Metric, model_sums: np.ndarray, baseline_sum: float, count: float) -> np.ndarray:
    return np.array([metric.compare(s / count, baseline_sum / count) for s in model_sums], dtype=float)


@dataclass
class NestedPairs:
    """Every (validation window, later test split) pair: window size, distance, gap and weight."""

    k: np.ndarray
    d: np.ndarray
    gap: np.ndarray
    weights: np.ndarray
    n_windows: int
    n_windows_other_pick: int


def nested_pairs(metric: Metric, scores: SplitScores, final_index: int, gap_splits: int = 0) -> NestedPairs:
    """The gap between the final model and each window's pick, on every split from gap_splits after the window on."""
    n = len(scores.counts)
    cum_models = np.vstack([np.zeros(scores.model_sums.shape[1]), np.cumsum(scores.model_sums, 0)])
    cum_baseline = np.concatenate([[0.0], np.cumsum(scores.baseline_sums)])
    cum_counts = np.concatenate([[0.0], np.cumsum(scores.counts)])
    per_split = np.array(
        [_comparisons(metric, scores.model_sums[t], scores.baseline_sums[t], scores.counts[t]) for t in range(n)]
    )
    ks, ds, gaps, weights = [], [], [], []
    n_windows = other = 0
    for a in range(n - 1 - gap_splits):
        for b in range(a + 1, n - gap_splits):
            validation = _comparisons(
                metric, cum_models[b] - cum_models[a], cum_baseline[b] - cum_baseline[a], cum_counts[b] - cum_counts[a]
            )
            pick = int(np.nanargmax(validation))
            n_windows += 1
            other += pick != final_index
            first = b + gap_splits
            ks.append(np.full(n - first, b - a))
            ds.append(np.arange(n - first))
            gaps.append(per_split[first:, final_index] - per_split[first:, pick])
            weights.append(scores.counts[first:])
    k, d, g, w = (np.concatenate(x) for x in (ks, ds, gaps, weights))
    keep = ~np.isnan(g)
    return NestedPairs(k[keep], d[keep], g[keep], w[keep].astype(float), n_windows, int(other))


def fit_gap(k: np.ndarray, gap: np.ndarray, weights: np.ndarray) -> float:
    """Weighted least squares of the gap on 1/sqrt(k), through the origin."""
    x = 1 / np.sqrt(k)
    return float(np.sum(weights * x * gap) / np.sum(weights * x * x))


def _points(x: np.ndarray, value: np.ndarray, weights: np.ndarray, line) -> list[SelectionBiasPoint]:
    return [
        SelectionBiasPoint(
            x=int(v),
            value=float(np.average(value[x == v], weights=weights[x == v])),
            fitted=float(line(v)),
            n_pairs=int((x == v).sum()),
        )
        for v in np.unique(x)
    ]


@dataclass
class SelectionBiasEstimate:
    final_index: int
    final_values: np.ndarray
    estimated_bias: float
    mean_gap: float
    n_pairs: int
    n_windows: int
    n_windows_other_pick: int
    inverse_sqrt_k: float
    window_size_points: list[SelectionBiasPoint]
    distance_points: list[SelectionBiasPoint]


def estimate_selection_bias(metric: Metric, scores: SplitScores, gap_splits: int = 0) -> SelectionBiasEstimate:
    """Run the nested scheme on per-split scores and estimate the final pick's bias at the full window size.

    gap_splits splits are left out between each validation window and its first test split.
    """
    n = len(scores.counts)
    if n < gap_splits + 3:
        raise SelectionBiasError(
            f"Need at least {gap_splits + 3} split points with scores for every model, got {n}"
            + (
                f" (the test splits start {gap_splits} splits after each window, so they share no target period)"
                if gap_splits
                else ""
            )
        )
    final_values = _comparisons(metric, scores.model_sums.sum(0), scores.baseline_sums.sum(), scores.counts.sum())
    final_index = int(np.nanargmax(final_values))
    pairs = nested_pairs(metric, scores, final_index, gap_splits)
    c = fit_gap(pairs.k, pairs.gap, pairs.weights)
    return SelectionBiasEstimate(
        final_index=final_index,
        final_values=final_values,
        estimated_bias=c / np.sqrt(n),
        mean_gap=float(np.average(pairs.gap, weights=pairs.weights)),
        n_pairs=len(pairs.gap),
        n_windows=pairs.n_windows,
        n_windows_other_pick=pairs.n_windows_other_pick,
        inverse_sqrt_k=c,
        window_size_points=_points(pairs.k, pairs.gap, pairs.weights, lambda kk: c / np.sqrt(kk)),
        distance_points=_points(
            pairs.d, pairs.gap - c / np.sqrt(pairs.k) + c / np.sqrt(n), pairs.weights, lambda _: c / np.sqrt(n)
        ),
    )


def period_ids(periods: pd.Series) -> pd.Series:
    """Period strings in one format ('2022-08' and '202208' both become '202208'), so observations and forecasts match."""
    ids = {period: TimePeriod.parse(str(period)).id for period in periods.unique()}
    return periods.map(ids)


def split_scores(
    metric: Metric,
    observations: pd.DataFrame,
    model_forecasts: list[pd.DataFrame],
    horizon_distance: int | None = None,
) -> tuple[list[str], list[int], SplitScores]:
    """Per-split metric sums of each model (the last one the baseline) on the forecasts all of them made.

    Scores every horizon distance, or only horizon_distance when it is given. Returns the split
    periods, the horizon distances scored and the sums.
    """
    observed = observations[observations.disease_cases.notna()]
    observed = FlatObserved.validate(observed.assign(time_period=period_ids(observed.time_period)))
    detailed = []
    for i, forecasts in enumerate(model_forecasts):
        if horizon_distance is not None:
            forecasts = forecasts[forecasts.horizon_distance == horizon_distance]
        forecasts = forecasts.assign(time_period=period_ids(forecasts.time_period))
        if forecasts.empty:
            raise SelectionBiasError(
                "No forecasts" + (f" at horizon distance {horizon_distance}" if horizon_distance is not None else "")
            )
        scored = metric.get_detailed_metric(observed, FlatForecasts.validate(forecasts))
        detailed.append(scored[[*KEYS, "metric"]].rename(columns={"metric": i}))
    common = reduce(lambda left, right: left.merge(right, on=KEYS), detailed).dropna()
    common["origin"] = origin_periods(common)
    sums = common.groupby("origin")[list(range(len(model_forecasts)))].sum().sort_index()
    counts = common.groupby("origin").size().reindex(sums.index).to_numpy(float)
    values = sums.to_numpy(float)
    horizons = sorted(int(h) for h in common.horizon_distance.unique())
    return (
        list(sums.index),
        horizons,
        SplitScores(model_sums=values[:, :-1], baseline_sums=values[:, -1], counts=counts),
    )


def _candidate_backtests(session: Session, specification_id: int) -> list[Backtest]:
    """The newest backtest of each model on the leaderboard: no baselines, no archived models."""
    backtests = session.exec(
        select(Backtest)
        .where(
            Backtest.specification_id == specification_id, is_not_baseline_backtest(), is_not_archived_model_backtest()
        )
        .order_by(col(Backtest.created).desc().nulls_last(), col(Backtest.id).desc())
    ).all()
    newest: dict[int, Backtest] = {}
    for backtest in backtests:
        newest.setdefault(backtest.model_db_id, backtest)
    return list(newest.values())


def get_selection_bias(
    session: Session,
    specification_id: int,
    metric_id: str,
    baseline_name: str | None = None,
    horizon_distance: int | None = None,
) -> SelectionBias:
    """Estimate the selection bias of the best model on a specification from all its split points.

    Scores every horizon distance unless horizon_distance picks one.

    Raises:
        SpecificationNotFoundError: the specification does not exist.
        SelectionBiasError: unknown metric or baseline, metric without a mean aggregation or a comparison,
            the baseline has not run, fewer than two candidate models, or too few split points
            (three, plus the gap between window and test splits).
    """
    if session.get(BacktestSpecification, specification_id) is None:
        raise SpecificationNotFoundError(f"Backtest specification {specification_id} not found")
    metric_cls = get_metric(metric_id)
    if metric_cls is None:
        raise SelectionBiasError(f"Unknown metric '{metric_id}'")
    if metric_cls.spec.comparison_op is None or metric_cls.spec.aggregation_op != AggregationOp.MEAN:
        raise SelectionBiasError(f"Metric '{metric_id}' needs a comparison operator and a mean aggregation")
    metric = metric_cls()
    if baseline_name is None:
        baseline_name = SettingsManager(session).get().default_baseline_model
    baselines = {model.name: model for model in SessionWrapper(session=session).get_baseline_models()}
    if baseline_name not in baselines:
        raise SelectionBiasError(f"Unknown baseline model '{baseline_name}'. Available: {sorted(baselines)}")
    baseline_backtest = session.exec(
        select(Backtest)
        .join(SpecificationBaseline, col(SpecificationBaseline.backtest_id) == Backtest.id)
        .where(
            SpecificationBaseline.specification_id == specification_id,
            SpecificationBaseline.configured_model_id == baselines[baseline_name].id,
            SpecificationBaseline.status == SpecificationBaselineStatus.done,
        )
    ).first()
    if baseline_backtest is None:
        raise SelectionBiasError(f"The baseline '{baseline_name}' has not run on this specification yet")
    candidates = _candidate_backtests(session, specification_id)
    if len(candidates) < 2:
        raise SelectionBiasError(f"Need at least 2 models on the specification, found {len(candidates)}")

    flats = [Evaluation.from_backtest(backtest).to_flat() for backtest in [*candidates, baseline_backtest]]
    periods, horizons, scores = split_scores(
        metric, flats[0].observations, [flat.forecasts for flat in flats], horizon_distance
    )
    gap_splits = horizons[-1] - horizons[0]
    estimate = estimate_selection_bias(metric, scores, gap_splits)
    final = candidates[estimate.final_index]
    validation_value = float(estimate.final_values[estimate.final_index])
    ranked = sorted(
        zip(candidates, estimate.final_values, strict=True), key=lambda pair: -np.nan_to_num(pair[1], nan=-np.inf)
    )
    return SelectionBias(
        metric_id=metric_id,
        baseline_model=baseline_name,
        comparison_description=get_comparison_description(metric_cls.spec.comparison_op),
        horizon_distances=horizons,
        gap_splits=gap_splits,
        split_periods=periods,
        candidates=[
            SelectionBiasCandidate(
                backtest_id=backtest.id,
                configured_model_id=backtest.model_db_id,
                model_name=backtest.model_id,
                validation_value=float(value),
            )
            for backtest, value in ranked
        ],
        selected_backtest_id=final.id,
        validation_value=validation_value,
        estimated_bias=estimate.estimated_bias,
        estimated_value=validation_value - estimate.estimated_bias,
        mean_gap=estimate.mean_gap,
        n_pairs=estimate.n_pairs,
        n_windows=estimate.n_windows,
        n_windows_other_pick=estimate.n_windows_other_pick,
        coefficients=SelectionBiasCoefficients(inverse_sqrt_k=estimate.inverse_sqrt_k),
        window_size_points=estimate.window_size_points,
        distance_points=estimate.distance_points,
    )

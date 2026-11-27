"""Selection bias of picking the best model on a backtest specification, estimated from its own split points.

Picking the model with the best score over the splits flatters that model: part of its lead
is luck, and it is gone on new data. The bias is estimated with a bootstrap over the split
points, after BBC-CV (Tsamardinos, Greasidou and Borboudakis 2018, Machine Learning 107). No
model is run again and nothing is fitted. One draw takes as many split points as there are,
with replacement, and picks the model with the best comparison with the baseline over the
drawn ones, a split drawn twice counting twice. The pick is then read on the splits the draw
left out, which it was not picked on, next to the final model f (the one picked on all splits):

    gap = value of f on the left-out splits - value of the draw's pick on the left-out splits

The gap is zero whenever the draw picks f, so a model that wins every draw has no selection
bias, however its score moves over time. The final model was picked with the left-out splits
included, so a draw that picks another model shows how much of f's lead on them was luck. The
estimated bias is the mean gap over the draws, and the estimated value on new data is the
final model's validation value minus it.

The paper's estimate is the mean of the picks' left-out values, and its bias the validation
value minus that. That is the same number when every split is left out equally often. Here
both values are read on the same left-out splits, so which splits a draw happens to leave out
does not move the estimate.

The paper draws single, independent observations. Split points are not independent, so two
things are added. Splits are drawn in blocks of consecutive ones (a circular block bootstrap),
so a model's good or bad stretch stays together. And with several horizon distances, the
forecasts of neighbouring split points share target periods, so a left-out split that shares a
month with a drawn one would not be independent of the pick: a left-out split is only scored
when it is more than (largest - smallest horizon distance) splits from every drawn one. That
gap assumes consecutive split points (stride 1). Without it the estimated bias comes out too
low. With it, few draws have a split left to score when the split points are few, and when
fewer than half of them have one, no estimate is given.

The estimate of one specification is rough, and it is too low when models do well or badly
together for longer stretches than a block.

Scores pool org units and, by default, every horizon distance, so the final model and its
validation value are the ones the leaderboard ranks on. Only forecasts every candidate and the
baseline made are scored, so all models are compared on the same forecasts.
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


class SelectionBiasCandidate(DBModel):
    """A candidate model, its comparison with the baseline over all split points and how often the draws pick it."""

    backtest_id: int = Field(description="Id of the candidate's backtest.")
    configured_model_id: int = Field(description="Id of the candidate's configured model.")
    model_name: str = Field(description="Name of the candidate's configured model.")
    validation_value: float = Field(description="Comparison with the baseline over all split points.")
    n_draws_picked: int = Field(
        description="Number of scored draws in which this model was the best on the drawn split points."
    )


class SelectionBias(DBModel):
    """Estimated selection bias of the best model on a specification, from a bootstrap over its split points."""

    metric_id: str = Field(description="Id of the metric that was compared with the baseline.")
    baseline_model: str = Field(description="Name of the baseline model the values are compared with.")
    comparison_description: str = Field(description="One-line description of what the comparison values mean.")
    horizon_distances: list[int] = Field(description="Horizon distances whose forecasts were scored.")
    gap_splits: int = Field(
        description="A left-out split point is only scored when it is more than this many splits from every "
        "drawn one, so the two share no target period: the largest minus the smallest horizon distance scored."
    )
    split_periods: list[str] = Field(description="Split points used, in order: the periods each split predicted from.")
    candidates: list[SelectionBiasCandidate] = Field(description="Every candidate model, best first.")
    selected_backtest_id: int = Field(description="Backtest of the model picked on all split points.")
    validation_value: float = Field(
        description="The picked model's comparison with the baseline over all split points."
    )
    estimated_bias: float = Field(
        description="How much of the picked model's lead is expected to be luck: the mean, over the scored draws, "
        "of the picked model's value minus the draw's pick's value on the split points the draw left out. Zero "
        "when every draw picks the same model. Too low when models do well or badly for longer stretches than "
        "blockSize split points."
    )
    estimated_value: float = Field(description="Estimated comparison on new data: validationValue - estimatedBias.")
    block_size: int = Field(description="Number of consecutive split points drawn together.")
    n_draws: int = Field(description="Number of draws.")
    n_draws_scored: int = Field(
        description="Draws that left out a split point more than gapSplits from every drawn one. Only these are used."
    )


@dataclass
class SplitScores:
    """Per-split sums of a mean-aggregated metric, on the forecasts every model made."""

    model_sums: np.ndarray  # (splits, models)
    baseline_sums: np.ndarray  # (splits,)
    counts: np.ndarray  # (splits,), forecasts scored


def _comparisons(metric: Metric, model_sums: np.ndarray, baseline_sum: float, count: float) -> np.ndarray:
    return np.array([metric.compare(s / count, baseline_sum / count) for s in model_sums], dtype=float)


N_DRAWS = 2000
MIN_BLOCK_SIZE = 3
MIN_SCORED_SHARE = 0.5


def left_out_splits(drawn: np.ndarray, gap_splits: int = 0) -> np.ndarray:
    """The splits a draw is scored on: not drawn, and more than gap_splits from every drawn split.

    drawn is a boolean array with the splits, in time order, on its last axis.
    """
    near = drawn.copy()
    for shift in range(1, gap_splits + 1):
        near[..., shift:] |= drawn[..., :-shift]
        near[..., :-shift] |= drawn[..., shift:]
    return ~near


@dataclass
class SelectionBiasEstimate:
    final_index: int
    final_values: np.ndarray
    estimated_bias: float
    block_size: int
    n_draws: int
    n_draws_scored: int
    n_draws_picked: np.ndarray  # (models,), scored draws that pick each model


def estimate_selection_bias(
    metric: Metric, scores: SplitScores, gap_splits: int = 0, n_draws: int = N_DRAWS, seed: int = 0
) -> SelectionBiasEstimate:
    """Estimate the final pick's bias with a block bootstrap over the split points.

    Each draw picks the best model on the drawn splits and is scored on the splits it left out,
    more than gap_splits from every drawn one. Blocks are max(3, gap_splits + 1) consecutive
    splits. The seed is fixed, so the same scores give the same estimate.

    Raises:
        SelectionBiasError: fewer than half of the draws leave out a split to score.
    """
    n, n_models = scores.model_sums.shape
    block_size = max(MIN_BLOCK_SIZE, gap_splits + 1)
    final_values = _comparisons(metric, scores.model_sums.sum(0), scores.baseline_sums.sum(), scores.counts.sum())
    final_index = int(np.nanargmax(final_values))

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_draws, -(-n // block_size)))
    splits = ((starts[:, :, None] + np.arange(block_size)) % n).reshape(n_draws, -1)[:, :n]
    times_drawn = np.zeros((n_draws, n))
    np.add.at(times_drawn, (np.arange(n_draws)[:, None], splits), 1)
    left_out = left_out_splits(times_drawn > 0, gap_splits)
    scored = left_out.any(1)
    if scored.sum() < MIN_SCORED_SHARE * n_draws:
        raise SelectionBiasError(
            f"Too few split points ({n}): only {scored.mean():.0%} of the draws leave out a split point to score, "
            "and at least half must"
            + (
                f". A left-out split point is only scored when it is more than {gap_splits} splits from every "
                "drawn one, so they share no target period: use more split points, or one horizon distance"
                if gap_splits
                else ""
            )
        )

    gaps = []  # one per scored draw: zero when the draw picks the final model
    picked = np.zeros(n_models, dtype=int)
    for times, test in zip(times_drawn[scored], left_out[scored], strict=True):
        drawn_values = _comparisons(
            metric, times @ scores.model_sums, times @ scores.baseline_sums, times @ scores.counts
        )
        pick = int(np.nanargmax(drawn_values))
        picked[pick] += 1
        final_value, pick_value = _comparisons(
            metric,
            scores.model_sums[test][:, [final_index, pick]].sum(0),
            scores.baseline_sums[test].sum(),
            scores.counts[test].sum(),
        )
        gaps.append(0.0 if pick == final_index else final_value - pick_value)
    return SelectionBiasEstimate(
        final_index=final_index,
        final_values=final_values,
        estimated_bias=float(np.nanmean(gaps)),
        block_size=block_size,
        n_draws=n_draws,
        n_draws_scored=int(scored.sum()),
        n_draws_picked=picked,
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
            the baseline has not run, fewer than two candidate models, or too few split points for the
            gap between drawn and left-out splits.
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
        zip(candidates, estimate.final_values, estimate.n_draws_picked, strict=True),
        key=lambda row: -np.nan_to_num(row[1], nan=-np.inf),
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
                n_draws_picked=int(n_picked),
            )
            for backtest, value, n_picked in ranked
        ],
        selected_backtest_id=final.id,
        validation_value=validation_value,
        estimated_bias=estimate.estimated_bias,
        estimated_value=validation_value - estimate.estimated_bias,
        block_size=estimate.block_size,
        n_draws=estimate.n_draws,
        n_draws_scored=estimate.n_draws_scored,
    )

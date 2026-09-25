from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

from chap_core.api_types import BacktestParams
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.hpo.hyperparameter_optimizer import HyperparameterOptimizer
from chap_core.hpo.searcher import GridSearcher, RandomSearcher, TPESearcher
from chap_core.hpo.types import SearchCandidate
from chap_core.models.model_template import ModelTemplate


class FakeObjective:
    """Small objective that can return scores, raise failures, or compute from config."""

    def __init__(
        self,
        scores: list[float | Exception] | None = None,
        *,
        direction: str = "minimize",
        score_fn: Callable[[ModelConfiguration], float] | None = None,
        metric: str = "rmse",
        backtest_params: BacktestParams | None = None,
    ) -> None:
        self._scores = iter(scores or [])
        self._score_fn = score_fn
        self.direction = SimpleNamespace(value=direction)
        self.metric = metric
        self.backtest_params = backtest_params or BacktestParams(
            n_periods=1,
            n_splits=1,
            stride=1,
        )
        self.model_template = cast(
            ModelTemplate,
            SimpleNamespace(
                model_template_config=SimpleNamespace(
                    name="fake-template",
                    version="1.0",
                )
            ),
        )
        self.configurations: list[ModelConfiguration] = []

    def __call__(self, model_configuration: ModelConfiguration, dataset: Any) -> float:
        del dataset
        self.configurations.append(model_configuration.model_copy(deep=True))
        if self._score_fn is not None:
            return float(self._score_fn(model_configuration))

        value = next(self._scores)
        if isinstance(value, Exception):
            raise value
        return float(value)


class FakeSearcher:
    """Deterministic searcher that records reset and tell interactions."""

    def __init__(self, params: list[dict[str, Any]]) -> None:
        self._params = params
        self._index = 0
        self.tell_calls: list[tuple[SearchCandidate, float | None]] = []
        self.reset_calls: list[tuple[dict[str, Any], int | None]] = []

    def reset(self, search_space: dict[str, Any], seed: int | None) -> None:
        self._index = 0
        self.tell_calls.clear()
        self.reset_calls.append((deepcopy(search_space), seed))

    def ask(self) -> SearchCandidate | None:
        if self._index >= len(self._params):
            return None

        candidate = SearchCandidate(
            params=dict(self._params[self._index]),
            token=self._index,
        )
        self._index += 1
        return candidate

    def tell(self, candidate: SearchCandidate, result: float | None) -> None:
        self.tell_calls.append((candidate, result))


def make_optimizer(
    objective: FakeObjective,
    searcher: Any,
    *,
    model_configuration: ModelConfiguration | None = None,
    search_space: dict[str, Any] | None = None,
    max_trials: int | None = None,
    seed: int | None = 123,
) -> HyperparameterOptimizer:
    return HyperparameterOptimizer(
        objective=objective,  # type: ignore[arg-type]
        searcher=searcher,  # type: ignore[arg-type]
        model_configuration=model_configuration,
        search_space=search_space or {"x": [1, 2]},
        max_trials=max_trials,
        seed=seed,
    )


@pytest.mark.parametrize(
    "non_finite_score",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_non_finite_score_is_failed_and_cannot_be_best(non_finite_score: float) -> None:
    """A NaN/inf objective result is a failed trial and must never become the best trial."""
    searcher = FakeSearcher([{"x": 1}, {"x": 2}])
    objective = FakeObjective([non_finite_score, 1.5])
    optimizer = make_optimizer(objective, searcher)

    result = optimizer.meta_learn(object())  # type: ignore[arg-type]

    assert result.best_params == {"x": 2}
    assert result.best_score == 1.5

    successful_trial = result.leaderboard[0]
    assert successful_trial.params == {"x": 2}
    assert successful_trial.score == 1.5
    assert successful_trial.failure is None

    failed_trial = result.leaderboard[1]
    assert failed_trial.params == {"x": 1}
    assert failed_trial.score is None
    assert failed_trial.failure is not None
    assert "Objective returned non-finite score" in failed_trial.failure

    assert [score for _, score in searcher.tell_calls] == [None, 1.5]


@pytest.mark.parametrize(
    "non_finite_score",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_all_non_finite_scores_raise(non_finite_score: float) -> None:
    """An HPO run with no successful trial fails instead of returning meaningless metadata."""
    searcher = FakeSearcher([{"x": 1}, {"x": 2}])
    objective = FakeObjective([non_finite_score, non_finite_score])
    optimizer = make_optimizer(objective, searcher)

    with pytest.raises(
        ValueError,
        match="Hyperparameter optimization completed without any successful trials",
    ):
        optimizer.meta_learn(object())  # type: ignore[arg-type]

    assert [score for _, score in searcher.tell_calls] == [None, None]


@pytest.mark.parametrize(
    "searcher_factory",
    [
        pytest.param(RandomSearcher, id="random"),
        pytest.param(lambda: TPESearcher("minimize"), id="tpe"),
    ],
)
def test_non_exhaustive_searchers_require_max_trials(searcher_factory: Callable[[], Any]) -> None:
    """Random/TPE search cannot terminate by exhaustion, so construction requires a trial budget."""
    with pytest.raises(ValueError, match="max_trials must be specified for non-exhaustive searchers"):
        make_optimizer(FakeObjective([1.0]), searcher_factory(), max_trials=None)


def test_grid_searcher_allows_no_max_trials() -> None:
    """Grid search is exhaustive and may legitimately use max_trials=None."""
    optimizer = make_optimizer(FakeObjective([1.0]), GridSearcher(), max_trials=None)

    assert optimizer._max_trials is None


def test_grid_search_runs_to_exhaustion_and_orders_minimize_leaderboard() -> None:
    """Exhaustive grid search evaluates every point and ranks successful trials ascending for minimize."""
    objective = FakeObjective(
        direction="minimize",
        score_fn=lambda config: float((config.user_option_values or {})["x"]),
    )
    optimizer = make_optimizer(
        objective,
        GridSearcher(),
        search_space={"x": [3, 1, 2]},
        max_trials=None,
    )

    result = optimizer.meta_learn(object())  # type: ignore[arg-type]

    assert result.stop_reason == "search_exhausted"
    assert result.best_params == {"x": 1}
    assert result.best_score == 1.0
    assert [trial.score for trial in result.leaderboard] == [1.0, 2.0, 3.0]
    assert [trial.trial_nr for trial in result.leaderboard] == [1, 2, 0]


def test_maximize_leaderboard_is_sorted_descending() -> None:
    """A maximizing metric reverses leaderboard ordering without changing trial identity."""
    searcher = FakeSearcher([{"x": 1}, {"x": 2}, {"x": 3}])
    objective = FakeObjective([1.0, 3.0, 2.0], direction="maximize")
    optimizer = make_optimizer(objective, searcher)

    result = optimizer.meta_learn(object())  # type: ignore[arg-type]

    assert result.best_params == {"x": 2}
    assert result.best_score == 3.0
    assert [trial.score for trial in result.leaderboard] == [3.0, 2.0, 1.0]
    assert [trial.trial_nr for trial in result.leaderboard] == [1, 2, 0]


def test_objective_failure_is_reported_to_searcher_and_ranked_after_successes() -> None:
    """Exceptions are recorded as failed trials, sent to tell(None), and moved behind successes."""
    searcher = FakeSearcher([{"x": 1}, {"x": 2}])
    objective = FakeObjective([RuntimeError("model crashed"), 4.0])
    optimizer = make_optimizer(objective, searcher)

    result = optimizer.meta_learn(object())  # type: ignore[arg-type]

    assert [score for _, score in searcher.tell_calls] == [None, 4.0]
    assert result.leaderboard[0].params == {"x": 2}
    assert result.leaderboard[0].failure is None
    assert result.leaderboard[1].params == {"x": 1}
    assert result.leaderboard[1].score is None
    assert result.leaderboard[1].failure == "RuntimeError: model crashed"


def test_max_trials_stops_before_searcher_exhaustion() -> None:
    """The optimizer honors the configured budget even when the searcher has more candidates."""
    searcher = FakeSearcher([{"x": 1}, {"x": 2}, {"x": 3}])
    objective = FakeObjective([3.0, 2.0, 1.0])
    optimizer = make_optimizer(objective, searcher, max_trials=2, seed=77)

    result = optimizer.meta_learn(object())  # type: ignore[arg-type]

    assert result.stop_reason == "max_trials"
    assert len(result.leaderboard) == 2
    assert len(searcher.tell_calls) == 2
    assert searcher.reset_calls == [({"x": [1, 2]}, 77)]


def test_result_contains_current_hpo_metadata_fields() -> None:
    """The optimizer returns the model/backtest/metric metadata now stored directly on the HPO result."""
    backtest_params = BacktestParams(n_periods=2, n_splits=3, stride=1)
    objective = FakeObjective(
        [1.0],
        metric="rmse",
        backtest_params=backtest_params,
    )
    optimizer = make_optimizer(
        objective,
        FakeSearcher([{"x": 1}]),
    )

    result = optimizer.meta_learn(object())  # type: ignore[arg-type]

    assert result.searcher == "FakeSearcher"
    assert result.model_template_name == "fake-template"
    assert result.model_template_version == "1.0"
    assert result.backtest_params is backtest_params
    assert result.metric == "rmse"


def test_make_configuration_merges_hpo_values_and_preserves_covariates() -> None:
    """HPO overrides searched options while preserving unrelated options and continuous covariates."""
    base = ModelConfiguration(
        user_option_values={"fixed": 10, "x": 0},
        additional_continuous_covariates=["rainfall"],
    )
    optimizer = make_optimizer(
        FakeObjective([1.0]),
        FakeSearcher([{"x": 2}]),
        model_configuration=base,
    )

    merged = optimizer._make_configuration({"x": 2})

    assert merged.user_option_values == {"fixed": 10, "x": 2}
    assert merged.additional_continuous_covariates == ["rainfall"]
    assert base.user_option_values == {"fixed": 10, "x": 0}
    assert base.additional_continuous_covariates == ["rainfall"]


def test_optimizer_takes_ownership_copies_of_inputs() -> None:
    """Mutating caller-owned configuration/search-space objects after construction cannot affect HPO."""
    base = ModelConfiguration(
        user_option_values={"fixed": 10},
        additional_continuous_covariates=["rainfall"],
    )
    search_space = {"x": [1, 2]}
    optimizer = make_optimizer(
        FakeObjective([1.0]),
        FakeSearcher([{"x": 1}]),
        model_configuration=base,
        search_space=search_space,
    )

    search_space["x"].append(99)
    assert base.user_option_values is not None
    base.user_option_values["fixed"] = 999
    assert base.additional_continuous_covariates is not None
    base.additional_continuous_covariates.append("temperature")

    merged = optimizer._make_configuration({"x": 1})

    assert optimizer._search_space == {"x": [1, 2]}
    assert merged.user_option_values == {"fixed": 10, "x": 1}
    assert merged.additional_continuous_covariates == ["rainfall"]


def test_model_template_and_information_are_proxied_from_objective() -> None:
    """CHAP can access both the underlying ModelTemplate and its model information through the optimizer."""
    objective = FakeObjective([1.0])
    optimizer = make_optimizer(objective, FakeSearcher([{"x": 1}]))

    assert optimizer.model_template is objective.model_template
    assert optimizer.model_information is objective.model_template.model_template_config

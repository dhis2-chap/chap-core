from typing import Any

import optuna
import pytest

from chap_core.hpo.search_space import Float, Int
from chap_core.hpo.searcher import GridSearcher, RandomSearcher, TPESearcher
from chap_core.hpo.types import SearchCandidate


def test_grid_searcher_enumerates_cartesian_product_and_exhausts() -> None:
    """Grid search returns the full Cartesian product in deterministic input order, then None."""
    searcher = GridSearcher()
    searcher.reset({"a": [1, 2], "b": ["x", "y"]}, seed=123)

    candidates = [searcher.ask() for _ in range(4)]

    assert [candidate.params for candidate in candidates if candidate is not None] == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]
    assert searcher.ask() is None


def test_grid_searcher_requires_reset_before_ask() -> None:
    """Using grid search before initialization gives a clear lifecycle error."""
    with pytest.raises(RuntimeError, match="Call reset"):
        GridSearcher().ask()


def test_grid_searcher_rejects_non_list_search_space() -> None:
    """Grid search accepts only explicit categorical/list grids, not continuous distributions."""
    searcher = GridSearcher()

    with pytest.raises(ValueError, match="only supports list-based search spaces"):
        searcher.reset({"x": Int(1, 3)}, seed=None)


def test_random_searcher_is_reproducible_for_same_seed() -> None:
    """A fixed seed reproduces the same mixed categorical/int/float random-search sequence."""
    space = {
        "cat": ["a", "b", "c"],
        "integer": Int(2, 8, step=2),
        "continuous": Float(0.1, 0.9),
    }
    first = RandomSearcher()
    second = RandomSearcher()
    first.reset(space, seed=1234)
    second.reset(space, seed=1234)

    first_sequence = [first.ask().params for _ in range(20)]
    second_sequence = [second.ask().params for _ in range(20)]

    assert first_sequence == second_sequence


def test_random_searcher_respects_bounds_steps_and_categories() -> None:
    """Uniform random sampling never leaves declared domains and respects discrete step grids."""
    searcher = RandomSearcher()
    searcher.reset(
        {
            "cat": ["left", "right"],
            "integer": Int(2, 8, step=3),
            "float": Float(0.1, 0.5, step=0.2),
        },
        seed=7,
    )

    samples = [searcher.ask().params for _ in range(100)]

    assert all(sample["cat"] in {"left", "right"} for sample in samples)
    assert all(sample["integer"] in {2, 5, 8} for sample in samples)
    expected_float_values = (0.1, 0.3, 0.5)
    assert all(
        any(sample["float"] == pytest.approx(expected) for expected in expected_float_values) for sample in samples
    )


def test_random_log_sampling_stays_inside_positive_bounds() -> None:
    """Log-uniform float/int sampling remains positive and within the inclusive declared bounds."""
    searcher = RandomSearcher()
    searcher.reset(
        {
            "float": Float(1e-3, 1e2, log=True),
            "integer": Int(1, 100, log=True),
        },
        seed=99,
    )

    samples = [searcher.ask().params for _ in range(200)]

    assert all(1e-3 <= sample["float"] <= 1e2 for sample in samples)
    assert all(1 <= sample["integer"] <= 100 for sample in samples)


@pytest.mark.parametrize(
    ("search_space", "message"),
    [
        pytest.param({}, "non-empty dict", id="empty-space"),
        pytest.param({"x": []}, "must be non-empty", id="empty-categorical"),
        pytest.param({"x": Float(1.0, 1.0)}, "low < high", id="float-equal-bounds"),
        pytest.param({"x": Float(2.0, 1.0)}, "low < high", id="float-reversed-bounds"),
        pytest.param(
            {"x": Float(0.1, 1.0, step=0.1, log=True)},
            "step must be None",
            id="float-log-with-step",
        ),
        pytest.param(
            {"x": Float(0.0, 1.0, log=True)},
            "requires low, high > 0",
            id="float-log-nonpositive",
        ),
        pytest.param({"x": Float(0.0, 1.0, step=0.0)}, "step must be > 0", id="float-zero-step"),
        pytest.param({"x": Int(3, 2)}, "low <= high", id="int-reversed-bounds"),
        pytest.param({"x": Int(1, 3, step=0)}, "step must be >= 1", id="int-zero-step"),
        pytest.param(
            {"x": Int(1, 10, step=2, log=True)},
            "step must be 1",
            id="int-log-with-step",
        ),
        pytest.param(
            {"x": Int(0, 10, log=True)},
            "requires low, high > 0",
            id="int-log-nonpositive",
        ),
        pytest.param({"x": object()}, "Unsupported spec", id="unsupported-spec"),
    ],
)
def test_random_searcher_rejects_invalid_search_spaces(
    search_space: dict[str, Any],
    message: str,
) -> None:
    """Search-space validation rejects malformed domains before random sampling starts."""
    with pytest.raises(ValueError, match=message):
        RandomSearcher().reset(search_space, seed=1)


def test_tpe_rejects_invalid_direction() -> None:
    """TPE only accepts Optuna's supported minimize/maximize directions."""
    with pytest.raises(ValueError, match="Invalid optimization direction"):
        TPESearcher("sideways")


def test_tpe_requires_reset_before_ask() -> None:
    """TPE ask() fails clearly if a study has not yet been initialized."""
    with pytest.raises(RuntimeError, match="Call reset"):
        TPESearcher("minimize").ask()


def test_tpe_successful_tell_completes_pending_trial() -> None:
    """A successful objective result completes the matching Optuna trial and clears pending state."""
    searcher = TPESearcher("minimize")
    searcher.reset({"x": [1, 2]}, seed=4)
    candidate = searcher.ask()
    study = searcher._study

    searcher.tell(candidate, 1.25)

    assert study is not None
    assert searcher._pending == {}
    assert len(study.trials) == 1
    assert study.trials[0].state is optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == pytest.approx(1.25)


def test_tpe_failed_tell_marks_trial_failed() -> None:
    """tell(..., None) maps an HPO trial failure to Optuna's FAIL state rather than a numeric score."""
    searcher = TPESearcher("minimize")
    searcher.reset({"x": [1, 2]}, seed=4)
    candidate = searcher.ask()
    study = searcher._study

    searcher.tell(candidate, None)

    assert study is not None
    assert searcher._pending == {}
    assert study.trials[0].state is optuna.trial.TrialState.FAIL
    assert study.trials[0].value is None


def test_tpe_rejects_candidate_without_token() -> None:
    """TPE feedback must carry the token assigned by ask() so it can identify the pending Optuna trial."""
    searcher = TPESearcher("minimize")
    searcher.reset({"x": [1, 2]}, seed=4)

    with pytest.raises(ValueError, match="missing its trial token"):
        searcher.tell(SearchCandidate(params={"x": 1}), 1.0)


def test_tpe_rejects_unknown_or_already_consumed_token() -> None:
    """A token cannot be invented or told twice; both would corrupt Optuna trial accounting."""
    searcher = TPESearcher("minimize")
    searcher.reset({"x": [1, 2]}, seed=4)
    candidate = searcher.ask()
    searcher.tell(candidate, 1.0)

    with pytest.raises(KeyError, match="No pending TPE trial"):
        searcher.tell(candidate, 2.0)

    with pytest.raises(KeyError, match="No pending TPE trial 999"):
        searcher.tell(SearchCandidate(params={"x": 1}, token=999), 1.0)


def test_tpe_reset_discards_pending_trials_from_previous_study() -> None:
    """reset() starts a fresh study, so feedback from a pre-reset candidate must be rejected."""
    searcher = TPESearcher("minimize")
    searcher.reset({"x": [1, 2]}, seed=4)
    old_candidate = searcher.ask()

    searcher.reset({"x": [10, 20]}, seed=5)

    with pytest.raises(KeyError, match="No pending TPE trial"):
        searcher.tell(old_candidate, 1.0)

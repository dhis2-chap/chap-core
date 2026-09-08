import pytest

from chap_core.assessment.metrics.base import OptimizationDirection
from chap_core.hpo.hpoModel import HpoModel
from chap_core.hpo.searcher import GridSearcher, RandomSearcher, TPESearcher


class FakeObjective:
    """Minimal stand-in for Objective that scores a candidate from a lookup table."""

    def __init__(self, scores: dict[int, float]):
        self.direction = OptimizationDirection.MINIMIZE
        self._scores = scores

    def __call__(self, params: dict, dataset) -> float:
        return self._scores[params["x"]]


@pytest.mark.parametrize(
    "searcher",
    [
        RandomSearcher(),
        TPESearcher("minimize"),
    ],
)
def test_hpo_model_requires_max_trials_for_non_exhaustive_searcher(searcher):
    with pytest.raises(
        ValueError,
        match="max_trials must be specified for non-exhaustive searchers",
    ):
        HpoModel(
            objective=None,  # type: ignore[arg-type]
            searcher=searcher,
            configuration=None,
            search_space={},
            max_trials=None,
            seed=None,
        )


def test_hpo_model_allows_unlimited_grid_search():
    model = HpoModel(
        objective=FakeObjective({1: 3.0, 2: 1.0}),  # type: ignore[arg-type]
        searcher=GridSearcher(),
        configuration=None,
        search_space={"x": [1, 2]},
        max_trials=None,
        seed=None,
    )

    leaderboard = model.get_leaderboard("dataset")

    assert [entry["config"] for entry in leaderboard] == [{"x": 2}, {"x": 1}]

import pytest

from chap_core.hpo.hpoModel import HpoModel
from chap_core.hpo.searcher import RandomSearcher, TPESearcher, GridSearcher


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
        objective=None,  # type: ignore[arg-type]
        searcher=GridSearcher(),
        configuration=None,
        search_space={},
        max_trials=None,
        seed=None,
    )

    assert model._max_trials is None

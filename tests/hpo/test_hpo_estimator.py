from unittest.mock import MagicMock, patch

import pytest

from chap_core.api_types import BacktestParams, SearcherType
from chap_core.cli_endpoints._common import get_hpo_estimator
from chap_core.hpo.searcher import GridSearcher, RandomSearcher, TPESearcher


def _forwarded_searcher(searcher_inp):
    """Run get_hpo_estimator with heavy deps stubbed and return the searcher
    instance it forwards to HpoModel."""
    template = MagicMock(name="ModelTemplate")
    backtest_params = BacktestParams(n_periods=3, n_splits=2, stride=1)
    with (
        patch("chap_core.hpo.base.load_search_space_from_config", return_value={}),
        patch("chap_core.hpo.hpoModel.HpoModel") as hpo_mock,
    ):
        get_hpo_estimator(
            template=template,
            model_configuration_yaml=None,
            backtest_params=backtest_params,
            searcher_inp=searcher_inp,
        )
    return hpo_mock.call_args.kwargs["searcher"]


@pytest.mark.parametrize(
    "searcher_inp, expected_type",
    [
        (SearcherType.GRID, GridSearcher),
        (SearcherType.RANDOM, RandomSearcher),
        (SearcherType.TPE, TPESearcher),
    ],
)
def test_get_hpo_estimator_maps_searcher_type(searcher_inp, expected_type):
    assert isinstance(_forwarded_searcher(searcher_inp), expected_type)


def test_get_hpo_estimator_forwards_none_searcher():
    # None is forwarded unchanged so HpoModel applies its own default.
    assert _forwarded_searcher(None) is None


def test_get_hpo_estimator_rejects_unknown_searcher():
    template = MagicMock(name="ModelTemplate")
    backtest_params = BacktestParams(n_periods=3, n_splits=2, stride=1)
    with (
        patch("chap_core.hpo.base.load_search_space_from_config", return_value={}),
        patch("chap_core.hpo.hpoModel.HpoModel"),
        pytest.raises(ValueError, match="Unknown searcher"),
    ):
        get_hpo_estimator(
            template=template,
            model_configuration_yaml=None,
            backtest_params=backtest_params,
            searcher_inp="bogus",  # type: ignore[arg-type]  # not a valid SearcherType member
        )

from unittest.mock import patch, MagicMock

import pytest

from chap_core.assessment.prediction_evaluator import backtest
from chap_core.exceptions import ModelFailedException


def test_backtest_passes_stride_to_train_test_generator():
    mock_estimator = MagicMock()
    mock_data = MagicMock()

    with patch("chap_core.assessment.prediction_evaluator.train_test_generator") as mock_ttg:
        mock_ttg.return_value = (MagicMock(), iter([]))

        list(backtest(mock_estimator, mock_data, prediction_length=3, n_test_sets=4, stride=2))

        mock_ttg.assert_called_once_with(mock_data, 3, 4, stride=2, future_weather_provider=None)


def _splits(n):
    """Build ``n`` (historic, future, truth) split tuples with distinguishable historic data."""
    return [(f"historic{i}", f"future{i}", MagicMock()) for i in range(n)]


def test_backtest_trains_once_by_default():
    mock_estimator = MagicMock()

    with patch("chap_core.assessment.prediction_evaluator.train_test_generator") as mock_ttg:
        mock_ttg.return_value = ("train_set", iter(_splits(4)))

        list(backtest(mock_estimator, MagicMock(), prediction_length=3, n_test_sets=4, stride=1))

    assert mock_estimator.train.call_count == 1
    # Split 0 trains on the dedicated train_set, preserving the single-train behaviour.
    mock_estimator.train.assert_called_once_with("train_set")


def test_backtest_retrains_at_evenly_spaced_splits():
    mock_estimator = MagicMock()

    with patch("chap_core.assessment.prediction_evaluator.train_test_generator") as mock_ttg:
        mock_ttg.return_value = ("train_set", iter(_splits(4)))

        list(backtest(mock_estimator, MagicMock(), prediction_length=3, n_test_sets=4, stride=1, n_retrain=2))

    assert mock_estimator.train.call_count == 2
    trained_on = [call.args[0] for call in mock_estimator.train.call_args_list]
    # Split 0 uses train_set; the halfway retrain (split 2) uses its expanding historic window.
    assert trained_on == ["train_set", "historic2"]


def test_backtest_skips_failing_split_and_keeps_the_rest():
    """One split failing must not discard the splits that succeeded."""
    mock_estimator = MagicMock()
    predictor = mock_estimator.train.return_value
    predictor.predict.side_effect = [
        "prediction0",
        ModelFailedException("inla segfaulted"),
        "prediction2",
    ]
    splits = _splits(3)

    with patch("chap_core.assessment.prediction_evaluator.train_test_generator") as mock_ttg:
        mock_ttg.return_value = ("train_set", iter(splits))

        results = list(backtest(mock_estimator, MagicMock(), prediction_length=3, n_test_sets=3, stride=1))

    assert len(results) == 2
    merged_predictions = [split[2].merge.call_args.args[0] for split in splits if split[2].merge.called]
    assert merged_predictions == ["prediction0", "prediction2"]


def test_backtest_raises_when_every_split_fails():
    """If nothing was predicted at all, the failure must surface rather than yield an empty backtest."""
    mock_estimator = MagicMock()
    mock_estimator.train.return_value.predict.side_effect = ModelFailedException("inla segfaulted")

    with patch("chap_core.assessment.prediction_evaluator.train_test_generator") as mock_ttg:
        mock_ttg.return_value = ("train_set", iter(_splits(3)))

        with pytest.raises(ModelFailedException, match="All 3 evaluation splits failed"):
            list(backtest(mock_estimator, MagicMock(), prediction_length=3, n_test_sets=3, stride=1))


def test_backtest_does_not_swallow_training_failures():
    """A model that cannot train at all is fatal; there is nothing to fall back on."""
    mock_estimator = MagicMock()
    mock_estimator.train.side_effect = ModelFailedException("training blew up")

    with patch("chap_core.assessment.prediction_evaluator.train_test_generator") as mock_ttg:
        mock_ttg.return_value = ("train_set", iter(_splits(3)))

        with pytest.raises(ModelFailedException, match="training blew up"):
            list(backtest(mock_estimator, MagicMock(), prediction_length=3, n_test_sets=3, stride=1))

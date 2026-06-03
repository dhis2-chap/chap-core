from unittest.mock import patch, MagicMock

from chap_core.assessment.prediction_evaluator import backtest


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
        mock_ttg.return_value = (MagicMock(), iter(_splits(4)))

        list(backtest(mock_estimator, MagicMock(), prediction_length=3, n_test_sets=4, stride=1))

    assert mock_estimator.train.call_count == 1
    mock_estimator.train.assert_called_once_with("historic0")


def test_backtest_retrains_at_evenly_spaced_splits():
    mock_estimator = MagicMock()

    with patch("chap_core.assessment.prediction_evaluator.train_test_generator") as mock_ttg:
        mock_ttg.return_value = (MagicMock(), iter(_splits(4)))

        list(backtest(mock_estimator, MagicMock(), prediction_length=3, n_test_sets=4, stride=1, n_retrain=2))

    assert mock_estimator.train.call_count == 2
    trained_on = [call.args[0] for call in mock_estimator.train.call_args_list]
    assert trained_on == ["historic0", "historic2"]

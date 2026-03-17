from unittest.mock import patch, MagicMock

from chap_core.assessment.prediction_evaluator import backtest


def test_backtest_passes_stride_to_train_test_generator():
    mock_estimator = MagicMock()
    mock_data = MagicMock()

    with patch("chap_core.assessment.prediction_evaluator.train_test_generator") as mock_ttg:
        mock_ttg.return_value = (MagicMock(), iter([]))

        list(backtest(mock_estimator, mock_data, prediction_length=3, n_test_sets=4, stride=2))

        mock_ttg.assert_called_once_with(mock_data, 3, 4, stride=2, future_weather_provider=None)

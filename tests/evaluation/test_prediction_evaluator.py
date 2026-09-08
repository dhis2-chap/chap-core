from unittest.mock import patch, MagicMock

from chap_core.assessment.prediction_evaluator import backtest


def test_backtest_uses_n_test_sets_for_retraining():
    mock_estimator = MagicMock()
    mock_estimator.train.return_value = MagicMock()
    train_set = MagicMock()
    test_generator = iter([])

    with patch("chap_core.assessment.prediction_evaluator._retrain_split_indices") as mock_retrain_indices:
        mock_retrain_indices.return_value = {0}

        list(
            backtest(
                estimator=mock_estimator,
                train_set=train_set,
                test_generator=test_generator,
                n_test_sets=4,
                n_retrain=2,
            )
        )

        mock_retrain_indices.assert_called_once_with(4, 2)


def _splits(n):
    """Build ``n`` (historic, future, truth) split tuples with distinguishable historic data."""
    return [(f"historic{i}", f"future{i}", MagicMock()) for i in range(n)]


def test_backtest_trains_once_by_default():
    mock_estimator = MagicMock()
    train_set = "train_set"
    test_generator = iter(_splits(4))

    list(
        backtest(
            estimator=mock_estimator,
            train_set=train_set,
            test_generator=test_generator,
            n_test_sets=4,
        )
    )

    assert mock_estimator.train.call_count == 1
    # Split 0 trains on the dedicated train_set, preserving the single-train behaviour.
    mock_estimator.train.assert_called_once_with("train_set")


def test_backtest_retrains_at_evenly_spaced_splits():
    mock_estimator = MagicMock()
    train_set = "train_set"
    test_generator = iter(_splits(4))

    list(
        backtest(
            estimator=mock_estimator,
            train_set=train_set,
            test_generator=test_generator,
            n_test_sets=4,
            n_retrain=2,
        )
    )

    assert mock_estimator.train.call_count == 2
    trained_on = [call.args[0] for call in mock_estimator.train.call_args_list]
    # Split 0 uses train_set; the halfway retrain (split 2) uses its expanding historic window.
    assert trained_on == ["train_set", "historic2"]

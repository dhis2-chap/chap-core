import pandas as pd
import pytest

from chap_core.assessment.alert_policy_scoring import (
    binary_metrics,
    categorical_metrics,
    categories,
    label_cells,
)


@pytest.fixture
def labelled(alert_history, alert_observations, alert_forecasts, alert_levels):
    return label_cells(alert_history, alert_observations, alert_forecasts, alert_levels)


def test_label_cells_judges_every_level(labelled):
    by_cell = {(r.time_period, r.horizon_distance, r.level): (r.outbreak, r.alert) for r in labelled.itertuples()}
    assert by_cell == {
        ("2023-06", 1, 0): (True, True),
        ("2023-06", 1, 1): (False, False),
        ("2023-06", 2, 0): (True, False),
        ("2023-06", 2, 1): (False, False),
        ("2024-06", 1, 0): (False, True),
        ("2024-06", 1, 1): (False, True),
    }


def test_label_cells_drops_cells_without_a_threshold(alert_history, alert_observations, alert_forecasts, alert_levels):
    july = alert_forecasts.assign(time_period="2023-07")
    labelled = label_cells(alert_history, alert_observations, pd.concat([alert_forecasts, july]), alert_levels)
    assert "2023-07" not in set(labelled["time_period"])


def test_categories_take_the_most_severe_level(labelled):
    ranked = categories(labelled)
    by_cell = {(r.time_period, r.horizon_distance): (r.observed, r.predicted) for r in ranked.itertuples()}
    assert by_cell == {("2023-06", 1): (1, 1), ("2023-06", 2): (1, 0), ("2024-06", 1): (0, 2)}


def test_binary_metrics_per_level(labelled):
    monitor = labelled[labelled["level"] == 0]
    metrics = binary_metrics(monitor["outbreak"], monitor["alert"])
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.0
    assert metrics["ppv"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["balanced_accuracy"] == 0.25
    assert metrics["mcc"] == pytest.approx(-0.5)

    action = labelled[labelled["level"] == 1]
    metrics = binary_metrics(action["outbreak"], action["alert"])
    assert metrics["sensitivity"] is None
    assert metrics["specificity"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] is None
    assert metrics["mcc"] is None


def test_categorical_metrics(labelled):
    ranked = categories(labelled)
    metrics = categorical_metrics(ranked["observed"], ranked["predicted"], n_categories=3)
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["macro_f1"] == pytest.approx((0 + 2 / 3 + 0) / 3)
    assert metrics["weighted_kappa"] is not None


def test_weighted_kappa_is_none_when_every_cell_is_one_category():
    constant = pd.Series([0, 0, 0])
    assert categorical_metrics(constant, constant, n_categories=3)["weighted_kappa"] is None

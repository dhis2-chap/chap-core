import numpy as np
import pandas as pd
import pytest

from chap_core.assessment.outbreak_metrics import (
    BinaryConfusion,
    get_binary_outbreak_metrics,
    get_categorical_outbreak_metrics,
)
from chap_core.assessment.outbreak_metrics.scoring import categories, label_cells, score


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


def test_binary_metrics_from_confusion_counts():
    metrics = {metric_id: spec.compute for metric_id, spec in get_binary_outbreak_metrics().items()}
    confusion = BinaryConfusion(tp=1, fp=1, fn=1, tn=0)
    assert metrics["sensitivity"](confusion) == 0.5
    assert metrics["specificity"](confusion) == 0.0
    assert metrics["ppv"](confusion) == 0.5
    assert metrics["npv"](confusion) == 0.0
    assert metrics["f1"](confusion) == 0.5
    assert metrics["balanced_accuracy"](confusion) == 0.25
    assert metrics["mcc"](confusion) == pytest.approx(-0.5)


def test_binary_metrics_are_none_where_undefined():
    metrics = get_binary_outbreak_metrics()
    no_outbreaks = BinaryConfusion(tp=0, fp=1, fn=0, tn=2)
    assert metrics["sensitivity"].compute(no_outbreaks) is None
    assert metrics["balanced_accuracy"].compute(no_outbreaks) is None
    assert metrics["mcc"].compute(no_outbreaks) is None


def test_categorical_metrics_from_confusion_matrix():
    metrics = get_categorical_outbreak_metrics()
    # observed 1 -> predicted 1, observed 1 -> predicted 0, observed 0 -> predicted 2
    confusion = np.array([[0, 0, 1], [1, 1, 0], [0, 0, 0]])
    assert metrics["accuracy"].compute(confusion) == pytest.approx(1 / 3)
    assert metrics["macro_f1"].compute(confusion) == pytest.approx((0 + 2 / 3 + 0) / 3)
    assert metrics["weighted_kappa"].compute(confusion) == pytest.approx(-2 / 3)


def test_weighted_kappa_is_none_when_every_cell_is_one_category():
    confusion = np.array([[3, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert get_categorical_outbreak_metrics()["weighted_kappa"].compute(confusion) is None


def test_score_pools_every_cell_without_grouping(labelled):
    (overall,) = score(labelled, n_levels=2, group_by=[])
    assert overall.group == {}
    assert overall.n_cells == 3
    assert overall.levels[0]["sensitivity"] == 0.5
    assert overall.levels[1]["specificity"] == pytest.approx(2 / 3)
    assert overall.categorical["accuracy"] == pytest.approx(1 / 3)


def test_score_groups_by_the_requested_dimensions(labelled):
    rows = score(labelled, n_levels=2, group_by=["time_period", "horizon_distance"])
    assert [row.group for row in rows] == [
        {"time_period": "2023-06", "horizon_distance": 1},
        {"time_period": "2023-06", "horizon_distance": 2},
        {"time_period": "2024-06", "horizon_distance": 1},
    ]
    assert [row.n_cells for row in rows] == [1, 1, 1]
    assert rows[0].categorical["accuracy"] == 1.0

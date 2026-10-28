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
    return label_cells(pd.concat([alert_history, alert_observations]), alert_forecasts, alert_levels)


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
    observations = pd.concat([alert_history, alert_observations])
    labelled = label_cells(observations, pd.concat([alert_forecasts, july]), alert_levels)
    assert "2023-07" not in set(labelled["time_period"])


def test_label_cells_computes_thresholds_from_the_history_before_each_split(
    alert_history, alert_observations, alert_levels
):
    """The 2023-06 observation lifts the June mean, so a borderline forecast alerts for 2023 but not for 2024."""
    forecasts = pd.DataFrame(
        {
            "location": "A",
            "time_period": ["2023-06"] * 3 + ["2024-06"] * 3,
            "horizon_distance": 1,
            "sample": [0, 1, 2] * 2,
            "forecast": 100.5,
        }
    )
    labelled = label_cells(pd.concat([alert_history, alert_observations]), forecasts, alert_levels)
    monitor = labelled[labelled["level"] == 0].set_index("time_period")["alert"]
    assert monitor["2023-06"] and not monitor["2024-06"]


def test_label_cells_keeps_a_forecast_periods_own_observation_out_of_its_threshold(alert_levels):
    """Weekly ids are unpadded, so 2023W10 sorts before 2023W9 as a string but comes after it in time."""
    observations = pd.DataFrame(
        {
            "location": "A",
            "time_period": [f"{year}W10" for year in range(2019, 2024)],
            "disease_cases": [90.0, 95.0, 105.0, 110.0, 1000.0],
        }
    )
    forecasts = pd.DataFrame(
        {"location": "A", "time_period": "2023W10", "horizon_distance": 2, "sample": [0, 1, 2], "forecast": 150.0}
    )
    labelled = label_cells(observations, forecasts, alert_levels)
    assert labelled.set_index("level")["alert"].to_dict() == {0: True, 1: True}
    assert labelled["outbreak"].all()


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
    assert metrics["categorical_accuracy"].compute(confusion) == pytest.approx(1 / 3)
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
    assert overall.categorical["categorical_accuracy"] == pytest.approx(1 / 3)


def test_score_groups_by_the_requested_dimensions(labelled):
    rows = score(labelled, n_levels=2, group_by=["time_period", "horizon_distance"])
    assert [row.group for row in rows] == [
        {"time_period": "2023-06", "horizon_distance": 1},
        {"time_period": "2023-06", "horizon_distance": 2},
        {"time_period": "2024-06", "horizon_distance": 1},
    ]
    assert [row.n_cells for row in rows] == [1, 1, 1]
    assert rows[0].categorical["categorical_accuracy"] == 1.0

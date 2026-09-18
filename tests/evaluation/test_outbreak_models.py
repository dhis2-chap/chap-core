import pandas as pd
import pytest

from chap_core.assessment.outbreak import (
    get_outbreak_model,
    get_outbreak_models_registry,
    list_outbreak_models,
)
from chap_core.assessment.outbreak.base import restrict_to_origin
from chap_core.assessment.outbreak.persistence import PersistenceOutbreakModel
from chap_core.assessment.outbreak.threshold_model import ThresholdOutbreakModel
from chap_core.assessment.thresholds.params import SeasonalParams


def test_both_models_are_registered():
    registry = get_outbreak_models_registry()
    assert registry["threshold"] is ThresholdOutbreakModel
    assert registry["persistence_of_anomaly"] is PersistenceOutbreakModel
    assert get_outbreak_model("nonexistent") is None


def test_listed_models_carry_metadata():
    listed = {entry["id"]: entry for entry in list_outbreak_models()}
    assert listed["threshold"]["name"] == "Forecast exceedance"
    assert listed["persistence_of_anomaly"]["description"]


def test_multi_line_params_are_rejected():
    """A channel band draws several lines; an alert is one decision against one line."""
    with pytest.raises(ValueError, match="exactly one threshold line"):
        ThresholdOutbreakModel(threshold_params=SeasonalParams(type="seasonal", std_multiplier=[1.0, 2.0]))


def test_threshold_model_probability_is_the_sample_fraction(two_season_history, make_flat_forecasts):
    """June's channel sits near 111, so six of ten samples clear it."""
    forecasts = make_flat_forecasts("A", "2023-06", 1, [200.0] * 6 + [50.0] * 4)
    result = ThresholdOutbreakModel().alert_probabilities(two_season_history, ["2023-06"], forecasts=forecasts)
    assert len(result) == 1
    assert result.iloc[0]["probability"] == pytest.approx(0.6)


def test_threshold_model_requires_forecasts(two_season_history):
    with pytest.raises(ValueError, match="`forecasts` is required"):
        ThresholdOutbreakModel().alert_probabilities(two_season_history, ["2023-06"])


def test_persistence_alerts_when_the_origin_ran_hot(two_season_history):
    """Standing in June with June far above its channel, alert on November."""
    history = pd.concat(
        [two_season_history, pd.DataFrame([{"location": "A", "time_period": "2023-06", "disease_cases": 500.0}])],
        ignore_index=True,
    )
    result = PersistenceOutbreakModel().alert_probabilities(history, ["2023-11"], origin_period="2023-06")
    assert len(result) == 1
    assert result.iloc[0]["time_period"] == "2023-11"
    assert result.iloc[0]["probability"] == 1.0


def test_persistence_is_quiet_when_the_origin_was_normal(two_season_history):
    history = pd.concat(
        [two_season_history, pd.DataFrame([{"location": "A", "time_period": "2023-06", "disease_cases": 100.0}])],
        ignore_index=True,
    )
    result = PersistenceOutbreakModel().alert_probabilities(history, ["2023-11"], origin_period="2023-06")
    assert result.iloc[0]["probability"] == 0.0


def test_persistence_reports_the_horizon_to_each_target(two_season_history):
    """June to November is five months out, which horizon_diff counts as 6."""
    history = pd.concat(
        [two_season_history, pd.DataFrame([{"location": "A", "time_period": "2023-06", "disease_cases": 500.0}])],
        ignore_index=True,
    )
    result = PersistenceOutbreakModel().alert_probabilities(history, ["2023-11"], origin_period="2023-06")
    assert result.iloc[0]["horizon_distance"] == 6


def test_persistence_requires_an_origin_period(two_season_history):
    with pytest.raises(ValueError, match="`origin_period` is required"):
        PersistenceOutbreakModel().alert_probabilities(two_season_history, ["2023-11"])


def test_observations_after_the_origin_do_not_raise_the_channel(two_season_history):
    """The baseline stops at the origin, so a later spike cannot mask today's anomaly.

    Without the cutoff the 2024 spike would lift June's channel above 500 and the
    model would fall silent about a June that plainly ran hot.
    """
    history = pd.concat(
        [
            two_season_history,
            pd.DataFrame(
                [
                    {"location": "A", "time_period": "2023-06", "disease_cases": 500.0},
                    {"location": "A", "time_period": "2024-06", "disease_cases": 5000.0},
                ]
            ),
        ],
        ignore_index=True,
    )
    result = PersistenceOutbreakModel().alert_probabilities(history, ["2023-11"], origin_period="2023-06")
    assert result.iloc[0]["probability"] == 1.0


def test_restrict_to_origin_keeps_the_origin_itself(two_season_history):
    """Standing in June you may use every June up to and including this one."""
    restricted = restrict_to_origin(two_season_history, "2018-06")
    assert restricted["time_period"].max() == "2018-06"
    assert "2018-06" in set(restricted["time_period"])


def test_restrict_to_origin_passes_everything_through_when_unset(two_season_history):
    assert len(restrict_to_origin(two_season_history, None)) == len(two_season_history)


def test_thresholds_are_computed_per_target_period(two_season_history):
    """November's channel is far below June's, and each target gets its own."""
    channels = ThresholdOutbreakModel().thresholds(two_season_history, ["2023-06", "2023-11"])
    by_period = channels.set_index("time_period")["threshold"]
    assert by_period["2023-11"] < by_period["2023-06"]


def test_mismatched_params_are_rejected():
    """The registry pairs each strategy with its params model; a mismatch is caught up front."""
    from chap_core.assessment.thresholds.params import PercentileParams

    with pytest.raises(ValueError, match="takes SeasonalParams params"):
        ThresholdOutbreakModel(threshold_params=PercentileParams(type="percentile"))

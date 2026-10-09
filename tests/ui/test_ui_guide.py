import pandas as pd
import pytest

from chap_core.services.model_marketplace import MarketplaceModel
from chap_core.ui.guide import (
    answer_text,
    best_model,
    fit_reason,
    horizon_limits,
    issue_line,
    model_fit,
    period_type,
    summarize_dataset,
)


@pytest.fixture
def model(marketplace_model):
    return MarketplaceModel.model_validate(marketplace_model)


@pytest.fixture
def laos(data_path):
    return summarize_dataset(data_path / "laos_subset.csv")


@pytest.mark.parametrize(
    ("first_period", "expected"),
    [
        ("2010-01", "monthly"),
        ("2010W01", "weekly"),
        ("2010-W01", "weekly"),
        ("2003-12-29/2004-01-04", "weekly"),
        ("2010-02-01/2010-02-28", "monthly"),
        ("2010", None),
    ],
)
def test_period_type_is_read_from_the_periods(first_period, expected):
    assert period_type([first_period]) == expected


def test_a_dataset_is_summarized_for_the_guide(laos):
    assert laos.period_type == "monthly"
    assert laos.locations > 1
    assert {"rainfall", "mean_temperature", "population"} <= set(laos.covariates)
    assert laos.has_polygons


def test_weekly_data_written_as_date_ranges_is_weekly(data_path, model):
    nicaragua = summarize_dataset(data_path / "nicaragua_weekly_subset.csv")
    assert nicaragua.period_type == "weekly"
    assert nicaragua.unit == "week"
    assert model_fit(model, nicaragua) == "needs monthly data, and this data is weekly"


def test_a_model_fits_data_with_what_it_needs(model, laos):
    assert model_fit(model, laos) is None
    assert fit_reason(model, laos) == "Fits: monthly data, uses rainfall and mean temperature, all in your data"


def test_a_model_does_not_fit_data_without_a_required_covariate(model, laos):
    needs_humidity = model.model_copy(
        update={"covariates": model.covariates.model_copy(update={"required": ["humidity"]})}
    )
    assert model_fit(needs_humidity, laos) == "needs a `humidity` column, which this data does not have"


def test_a_monthly_model_does_not_fit_weekly_data(model, laos):
    weekly = laos.__class__(**{**laos.__dict__, "period_type": "weekly"})
    assert model_fit(model, weekly) == "needs monthly data, and this data is weekly"


def test_horizons_are_limited_by_every_chosen_model(model):
    short = model.model_copy(
        update={"compatibility": model.compatibility.model_copy(update={"max_prediction_periods": 3})}
    )
    assert horizon_limits([model, short]) == (1, 3)


def test_the_best_model_has_the_lowest_crps():
    metrics = pd.DataFrame({"model": ["A", "B"], "crps": [12.0, 9.5], "mae": [10.0, 11.0]})
    assert best_model(metrics)["model"] == "B"


def test_the_answer_says_when_another_model_came_closer_on_average():
    metrics = pd.DataFrame(
        {"model": ["A", "B"], "crps": [12.0, 9.5], "mae": [10.0, 11.0], "coverage_10_90": [0.6, 0.75]}
    )
    best = best_model(metrics)
    text = answer_text(best, metrics[metrics["model"] != "B"], "month")
    assert text.startswith("It has the best overall score, the CRPS: 9.5, against 12.0 for A.")
    assert "A (10) came closer on average, but B was more right about its uncertainty" in text


def test_a_validation_issue_names_where_it_is_and_the_periods():
    from chap_core.services.dataset_validation import ValidationIssue

    gap = ValidationIssue(level="error", message="Missing time periods", location="Bokeo", time_periods=["2010-11"])
    assert issue_line(gap) == "Missing time periods (Bokeo, 2010-11)"
    many = gap.model_copy(update={"time_periods": [f"2010-0{m}" for m in range(1, 8)]})
    assert issue_line(many).endswith("2010-05 and 2 more)")

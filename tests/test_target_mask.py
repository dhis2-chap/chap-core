import numpy as np
import pytest

from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.assessment.target_mask import (
    MASK_INDICATOR_FIELD,
    apply_target_mask,
    check_mask_outside_evaluation_window,
    as_future_data,
    check_target_column_allows_masking,
    without_target,
)
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.dataset_tables import DataSetCreateInfo, MaskedRange


def _range(dataset, first: int, last: int) -> list[MaskedRange]:
    periods = dataset.period_range
    return [MaskedRange(start=periods[first].id, end=periods[last].id)]


def test_mask_sets_nan_and_indicator_inside_the_range_only(health_population_data):
    masked = apply_target_mask(health_population_data, _range(health_population_data, 10, 12))
    for location, data in masked.items():
        original = health_population_data[location].disease_cases
        assert np.all(np.isnan(data.disease_cases[10:13]))
        assert list(data.disease_cases_masked[10:13]) == [1, 1, 1]
        assert np.array_equal(data.disease_cases[:10], original[:10], equal_nan=True)
        assert np.array_equal(data.disease_cases[13:], original[13:], equal_nan=True)
        assert not np.any(data.disease_cases_masked[:10]) and not np.any(data.disease_cases_masked[13:])


def test_mask_leaves_the_input_dataset_unchanged(health_population_data):
    location = next(iter(health_population_data.keys()))
    before = health_population_data[location].disease_cases.copy()
    apply_target_mask(health_population_data, _range(health_population_data, 10, 12))
    assert np.array_equal(health_population_data[location].disease_cases, before, equal_nan=True)
    assert MASK_INDICATOR_FIELD not in health_population_data.field_names()


def test_mask_is_idempotent(health_population_data):
    ranges = _range(health_population_data, 10, 12)
    once = apply_target_mask(health_population_data, ranges)
    twice = apply_target_mask(once, ranges)
    for location in once.keys():
        assert np.array_equal(once[location].disease_cases, twice[location].disease_cases, equal_nan=True)
        assert np.array_equal(once[location].disease_cases_masked, twice[location].disease_cases_masked)


def test_no_ranges_leaves_the_dataset_as_it_is(health_population_data):
    assert apply_target_mask(health_population_data, []) is health_population_data


def test_mask_rejects_ranges_of_another_period_type(health_population_data):
    with pytest.raises(ValueError, match="month range"):
        apply_target_mask(health_population_data, [MaskedRange(start="2023-06", end="2023-08")])


def test_mask_ending_before_the_evaluation_window_is_accepted(health_population_data):
    # n_periods=3, n_splits=2, stride=1 evaluates the last 4 periods; period -5 is the last training period.
    masked = apply_target_mask(health_population_data, _range(health_population_data, -7, -5))
    check_mask_outside_evaluation_window(masked, n_periods=3, n_splits=2, stride=1)


def test_mask_touching_the_evaluation_window_is_rejected(health_population_data):
    masked = apply_target_mask(health_population_data, _range(health_population_data, -6, -4))
    first_evaluation_period = health_population_data.period_range[-4].id
    with pytest.raises(ValueError, match=f"Masked period {first_evaluation_period} falls in the evaluation window"):
        check_mask_outside_evaluation_window(masked, n_periods=3, n_splits=2, stride=1)


def test_alternate_target_is_rejected_on_a_masked_dataset():
    ranges = [MaskedRange(start="2021-06", end="2021-08")]
    check_target_column_allows_masking(ranges, "disease_cases")
    check_target_column_allows_masking([], "rainfall")
    with pytest.raises(ValueError, match="'rainfall' cannot be used"):
        check_target_column_allows_masking(ranges, "rainfall")


def test_without_target_removes_target_and_indicator(health_population_data):
    masked = apply_target_mask(health_population_data, _range(health_population_data, 10, 12))
    assert without_target(masked).field_names() == ["population"]
    assert without_target(health_population_data).field_names() == ["population"]


@pytest.mark.parametrize("provider", ["climatology", "observed"])
def test_future_data_carries_an_all_zero_indicator(weekly_full_data, provider):
    masked = apply_target_mask(weekly_full_data, _range(weekly_full_data, 5, 7))
    _, splits = train_test_generator(masked, prediction_length=3, n_test_sets=1, future_weather_provider=provider)
    historic, future, _ = next(iter(splits))
    assert set(historic.field_names()) - {"disease_cases"} == set(future.field_names())
    assert all(not np.any(data.disease_cases_masked) for data in future.values())


def test_unmasked_future_data_has_no_indicator(weekly_full_data):
    _, splits = train_test_generator(weekly_full_data, prediction_length=3, n_test_sets=1)
    _, future, _ = next(iter(splits))
    assert MASK_INDICATOR_FIELD not in future.field_names()


def test_as_future_data_drops_the_target_and_zeroes_the_indicator(health_population_data):
    masked = apply_target_mask(health_population_data, _range(health_population_data, 10, 12))
    future = as_future_data(masked)
    assert future.field_names() == ["population", MASK_INDICATOR_FIELD]
    assert all(not np.any(data.disease_cases_masked) for data in future.values())
    assert as_future_data(health_population_data).field_names() == ["population"]


def test_a_masked_dataset_is_never_stored(health_population_data, engine):
    masked = apply_target_mask(health_population_data, _range(health_population_data, 10, 12))
    with SessionWrapper(engine) as session, pytest.raises(ValueError, match="never stored"):
        DataSetManager(session.session).save_dataset(DataSetCreateInfo(name="masked"), masked, None)


def test_dataset_apply_masked_ranges_matches_apply_target_mask(health_population_data):
    ranges = _range(health_population_data, 10, 12)
    via_method = health_population_data.apply_masked_ranges(ranges)
    via_function = apply_target_mask(health_population_data, ranges)
    for location in via_function.keys():
        assert np.array_equal(via_method[location].disease_cases, via_function[location].disease_cases, equal_nan=True)
        assert np.array_equal(via_method[location].disease_cases_masked, via_function[location].disease_cases_masked)
    assert health_population_data.apply_masked_ranges([]) is health_population_data


def test_mask_rejects_a_dataset_without_disease_cases(health_population_data):
    ranges = _range(health_population_data, 10, 12)
    with pytest.raises(ValueError, match="has no 'disease_cases' column"):
        apply_target_mask(health_population_data.remove_field("disease_cases"), ranges)

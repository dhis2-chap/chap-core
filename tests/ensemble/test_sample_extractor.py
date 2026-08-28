import numpy as np
import pytest

from chap_core.datatypes import Samples
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.ensemble import _meta_models
from chap_core.ensemble._sample_extractor import SampleExtractor


def _samples_from_weekly_data(weekly_full_data):
    location = next(iter(weekly_full_data.locations()))
    series = weekly_full_data[location]
    base = np.asarray(series.disease_cases, float)
    samples = np.vstack([base, base + 1.0, base + 2.0]).T
    return Samples(series.time_period, samples)


def _single_location_preds(weekly_full_data):
    """A DataSet of predictions plus the reference frame for the same single location."""
    location = next(iter(weekly_full_data.locations()))
    samples = _samples_from_weekly_data(weekly_full_data)
    preds_ds = DataSet({location: samples})
    df_all = weekly_full_data.to_pandas()
    df_ref = df_all[df_all["location"] == location][["location", "time_period"]].copy()
    return preds_ds, df_ref, samples


def test_reshape_samples_is_deterministic(weekly_full_data):
    preds_ds, df_ref, samples = _single_location_preds(weekly_full_data)
    target_n = samples.samples.shape[1] - 1

    first = SampleExtractor.reshape_samples(preds_ds, df_ref, target_n)
    second = SampleExtractor.reshape_samples(preds_ds, df_ref, target_n)

    assert np.allclose(first, second, equal_nan=True)


def test_reshape_samples_preserves_row_range(weekly_full_data):
    """Resampling interpolates each row's own quantiles, so no row leaves its own range."""
    preds_ds, df_ref, samples = _single_location_preds(weekly_full_data)

    for target_n in (2, 7):
        actual = SampleExtractor.reshape_samples(preds_ds, df_ref, target_n)
        assert actual.shape == (samples.samples.shape[0], target_n)
        row_min = samples.samples.min(axis=1)
        row_max = samples.samples.max(axis=1)
        finite = np.isfinite(row_min) & np.isfinite(row_max)
        assert finite.any()
        assert np.all(actual[finite] >= row_min[finite, None] - 1e-9)
        assert np.all(actual[finite] <= row_max[finite, None] + 1e-9)


def test_reshape_samples_row_order_fallback_checks_length(weekly_full_data):
    """Row-order alignment is only safe when the row counts actually match."""
    _preds_ds, df_ref, samples = _single_location_preds(weekly_full_data)
    # A bare Samples object carries no location column, so alignment falls back to row order.
    truncated = Samples(samples.time_period[:-1], samples.samples[:-1])

    with pytest.raises(ValueError, match="Cannot align predictions by row order"):
        SampleExtractor.reshape_samples(truncated, df_ref, 3)


def test_samples_to_flat_uses_median_for_samples(weekly_full_data):
    samples = _samples_from_weekly_data(weekly_full_data)
    location = next(iter(weekly_full_data.locations()))
    preds_ds = DataSet({location: samples})
    df_flat = SampleExtractor.samples_to_flat(preds_ds)

    series = weekly_full_data[location]
    base = np.asarray(series.disease_cases, float)
    expected = base + 1.0

    assert np.allclose(df_flat["forecast"].to_numpy(), expected, equal_nan=True)


def test_probabilistic_meta_model_fallback_on_failed_opt(monkeypatch, weekly_full_data):
    samples = _samples_from_weekly_data(weekly_full_data)
    base = samples.samples
    X_samples = [base[:, :2], base[:, 1:3]]
    y = base[:, 0]

    class FakeRes:
        success = False
        x = np.array([np.nan, np.nan])
        fun = np.nan

    def fake_minimize(*_args, **_kwargs):
        return FakeRes()

    monkeypatch.setattr(_meta_models, "minimize", fake_minimize)

    model = _meta_models.ProbabilisticMetaModel()
    model.fit(X_samples, y)

    assert np.allclose(model.coef_, np.array([0.5, 0.5]))


def test_reshape_samples_repeats_keyless_point_forecasts(weekly_full_data, point_forecast_only_factory):
    """The documented point-forecast fallback was unreachable: it called samples_to_flat,
    which raises on exactly the columns this branch is entered for."""
    _preds_ds, df_ref, _samples = _single_location_preds(weekly_full_data)
    values = np.arange(len(df_ref), dtype=float)

    actual = SampleExtractor.reshape_samples(point_forecast_only_factory(values), df_ref, 4)

    assert actual.shape == (len(df_ref), 4)
    np.testing.assert_allclose(actual, np.tile(values.reshape(-1, 1), (1, 4)))


def test_reshape_samples_keyless_point_forecasts_check_length(weekly_full_data, point_forecast_only_factory):
    _preds_ds, df_ref, _samples = _single_location_preds(weekly_full_data)
    values = np.arange(len(df_ref) - 1, dtype=float)

    with pytest.raises(ValueError, match="Cannot align predictions by row order"):
        SampleExtractor.reshape_samples(point_forecast_only_factory(values), df_ref, 4)

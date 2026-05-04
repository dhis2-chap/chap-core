import numpy as np
import pytest

from chap_core.ensemble._meta_models import NonNegativeMetaModel, ProbabilisticMetaModel


def _base_series_from_weekly_data(weekly_full_data):
    location = next(iter(weekly_full_data.locations()))
    series = weekly_full_data[location]
    base = np.asarray(series.disease_cases, float)
    return base[np.isfinite(base)]


def test_non_negative_meta_model_fits_and_predicts(weekly_full_data):
    base = _base_series_from_weekly_data(weekly_full_data)
    X = np.column_stack([base, base + 1.0])
    y = base + 0.5

    model = NonNegativeMetaModel().fit(X, y)
    preds = model.predict(X)

    assert model.coef_ is not None
    assert np.all(model.coef_ >= 0)
    assert preds.shape == y.shape


def test_probabilistic_meta_model_weights_on_simplex(weekly_full_data):
    base = _base_series_from_weekly_data(weekly_full_data)
    y = base
    x1 = np.tile(base.reshape(-1, 1), (1, 3))
    x2 = np.tile((base + 1.0).reshape(-1, 1), (1, 3))
    X_samples = [x1, x2]

    model = ProbabilisticMetaModel().fit(X_samples, y)

    assert model.coef_ is not None
    assert np.all(model.coef_ >= 0)
    assert np.isclose(float(np.sum(model.coef_)), 1.0, atol=1e-6)


def test_probabilistic_meta_model_rejects_shape_mismatch(weekly_full_data):
    base = _base_series_from_weekly_data(weekly_full_data)
    y = base
    x1 = np.tile(base.reshape(-1, 1), (1, 2))
    x2 = np.tile(base.reshape(-1, 1), (1, 3))

    with pytest.raises(ValueError, match="Sample shape mismatch"):
        ProbabilisticMetaModel().fit([x1, x2], y)

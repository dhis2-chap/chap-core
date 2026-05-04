import numpy as np

from chap_core.datatypes import Samples
from chap_core.ensemble import _meta_models
from chap_core.ensemble._sample_extractor import SampleExtractor


def _samples_from_weekly_data(weekly_full_data):
    location = next(iter(weekly_full_data.locations()))
    series = weekly_full_data[location]
    base = np.asarray(series.disease_cases, float)
    samples = np.vstack([base, base + 1.0, base + 2.0]).T
    return Samples(series.time_period, samples)


def test_reshape_samples_respects_rng(weekly_full_data):
    samples = _samples_from_weekly_data(weekly_full_data)
    df_ref = weekly_full_data.to_pandas()[["location", "time_period"]].copy()
    n_samp = samples.samples.shape[1]
    target_n = n_samp - 1

    rng_expected = np.random.default_rng(123)
    idx = rng_expected.choice(n_samp, target_n, replace=True)
    expected = samples.samples[:, idx]

    rng_call = np.random.default_rng(123)
    actual = SampleExtractor.reshape_samples(samples, df_ref, target_n, rng=rng_call)

    assert np.allclose(actual, expected, equal_nan=True)


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

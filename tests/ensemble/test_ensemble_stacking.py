import numpy as np

from chap_core.ensemble.ensemble_model import EnsembleModel


def test_deterministic_predict_shape_and_weights(weekly_full_data, constant_template_factory):
    templates = [
        constant_template_factory(5.0, 1, "model_a"),
        constant_template_factory(10.0, 1, "model_b"),
    ]
    model = EnsembleModel(base_templates=templates, method="deterministic", n_samples=5)

    predictor = model.train(weekly_full_data)
    preds = predictor.predict(weekly_full_data, weekly_full_data)

    assert model.weights is not None
    assert len(model.weights) == 2
    assert np.isclose(float(np.sum(model.weights)), 100.0, atol=1e-6)

    for loc in weekly_full_data.locations():
        samples = preds[loc].samples
        assert samples.shape[1] == 1
        assert samples.shape[0] == len(weekly_full_data[loc].time_period)


def test_deterministic_residual_bootstrap_generates_samples(weekly_full_data, constant_template_factory):
    templates = [
        constant_template_factory(2.0, 1, "model_a"),
        constant_template_factory(4.0, 1, "model_b"),
    ]
    n_samples = 4
    model = EnsembleModel(
        base_templates=templates,
        method="deterministic",
        n_samples=n_samples,
        use_residual_bootstrap=True,
        random_state=123,
    )

    predictor = model.train(weekly_full_data)
    preds = predictor.predict(weekly_full_data, weekly_full_data)

    for loc in weekly_full_data.locations():
        samples = preds[loc].samples
        assert samples.shape[1] == n_samples
        assert samples.shape[0] == len(weekly_full_data[loc].time_period)


def test_deterministic_residual_bootstrap_varies_samples(weekly_full_data, constant_template_factory):
    templates = [
        constant_template_factory(2.0, 1, "model_a"),
        constant_template_factory(4.0, 1, "model_b"),
    ]
    model = EnsembleModel(
        base_templates=templates,
        method="deterministic",
        n_samples=6,
        use_residual_bootstrap=True,
        random_state=123,
    )

    predictor = model.train(weekly_full_data)
    preds = predictor.predict(weekly_full_data, weekly_full_data)

    has_variation = False
    for loc in weekly_full_data.locations():
        samples = preds[loc].samples
        if np.unique(samples).size > 1:
            has_variation = True
            break

    assert has_variation


def test_probabilistic_predict_samples_count(weekly_full_data, constant_template_factory):
    templates = [
        constant_template_factory(3.0, 2, "model_a"),
        constant_template_factory(6.0, 2, "model_b"),
    ]
    n_samples = 6
    model = EnsembleModel(base_templates=templates, method="probabilistic", n_samples=n_samples, random_state=7)

    predictor = model.train(weekly_full_data)
    preds = predictor.predict(weekly_full_data, weekly_full_data)

    for loc in weekly_full_data.locations():
        samples = preds[loc].samples
        assert samples.shape[1] == n_samples
        assert samples.shape[0] == len(weekly_full_data[loc].time_period)

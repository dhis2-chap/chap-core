import numpy as np

from chap_core.ensemble._meta_models import NonNegativeMetaModel, ProbabilisticMetaModel
from chap_core.ensemble._predictor import EnsemblePredictor


class _FixedMetaDeterministic(NonNegativeMetaModel):
    def __init__(self, coef):
        super().__init__()
        self.coef_ = np.asarray(coef, float)

    def predict(self, X):
        coef = self.coef_
        assert coef is not None
        return np.dot(X, coef)


class _FixedMetaProbabilistic(ProbabilisticMetaModel):
    def __init__(self, coef):
        super().__init__()
        self.coef_ = np.asarray(coef, float)

    def predict(self, X_samples):
        coef = self.coef_
        assert coef is not None
        ens = sum(coef[i] * X_samples[i] for i in range(len(X_samples)))
        return np.maximum(ens, 0.0)


def test_predictor_deterministic_with_bootstrap(weekly_full_data, constant_predictor_factory, base_residuals_factory):
    predictors = [constant_predictor_factory(2.0, 1), constant_predictor_factory(4.0, 1)]
    meta = _FixedMetaDeterministic([0.25, 0.75])
    base_residuals = [
        base_residuals_factory(2.0),
        base_residuals_factory(4.0),
    ]

    predictor = EnsemblePredictor(
        predictors=predictors,
        meta=meta,
        probabilistic=False,
        n_samples=3,
        use_residual_bootstrap=True,
        base_residuals=base_residuals,
        rng=np.random.default_rng(123),
    )

    preds = predictor.predict(weekly_full_data, weekly_full_data)

    for loc in weekly_full_data.locations():
        samples = preds[loc].samples
        assert samples.shape[1] == 3
        assert samples.shape[0] == len(weekly_full_data[loc].time_period)


def test_predictor_deterministic_weight_fallback(weekly_full_data, constant_predictor_factory, base_residuals_factory):
    predictors = [constant_predictor_factory(2.0, 1), constant_predictor_factory(4.0, 1)]
    meta = _FixedMetaDeterministic([0.0, 0.0])
    base_residuals = [
        base_residuals_factory(2.0),
        base_residuals_factory(4.0),
    ]

    predictor = EnsemblePredictor(
        predictors=predictors,
        meta=meta,
        probabilistic=False,
        n_samples=2,
        use_residual_bootstrap=True,
        base_residuals=base_residuals,
        rng=np.random.default_rng(321),
    )

    preds = predictor.predict(weekly_full_data, weekly_full_data)

    for loc in weekly_full_data.locations():
        samples = preds[loc].samples
        assert samples.shape[1] == 2
        assert samples.shape[0] == len(weekly_full_data[loc].time_period)


def test_predictor_probabilistic_samples(weekly_full_data, constant_predictor_factory):
    predictors = [constant_predictor_factory(1.0, 2), constant_predictor_factory(3.0, 2)]
    meta = _FixedMetaProbabilistic([0.5, 0.5])

    predictor = EnsemblePredictor(
        predictors=predictors,
        meta=meta,
        probabilistic=True,
        n_samples=4,
        rng=np.random.default_rng(7),
    )

    preds = predictor.predict(weekly_full_data, weekly_full_data)

    for loc in weekly_full_data.locations():
        samples = preds[loc].samples
        assert samples.shape[1] == 4
        assert samples.shape[0] == len(weekly_full_data[loc].time_period)

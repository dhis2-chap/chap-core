import numpy as np
import pytest

from chap_core.datatypes import Samples
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
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


def test_predictor_deterministic_missing_rows_raises(weekly_full_data, constant_predictor_factory):
    class _MissingPredictor:
        def __init__(self, value: float):
            self._value = value

        def predict(self, _historic_data, future_data):
            result = {}
            for loc in future_data.locations():
                tp = future_data[loc].time_period
                if len(tp) <= 1:
                    tp_use = tp
                else:
                    tp_use = tp[:-1]
                vals = np.full(len(tp_use), self._value, dtype=float)
                samples = np.tile(vals.reshape(-1, 1), (1, 1))
                result[loc] = Samples(tp_use, samples)
            return DataSet(result)

    predictors = [_MissingPredictor(2.0), constant_predictor_factory(4.0, 1)]
    meta = _FixedMetaDeterministic([0.5, 0.5])

    predictor = EnsemblePredictor(
        predictors=predictors,
        meta=meta,
        probabilistic=False,
        n_samples=1,
        rng=np.random.default_rng(11),
    )

    with pytest.raises(ValueError, match="Missing base model predictions"):
        predictor.predict(weekly_full_data, weekly_full_data)

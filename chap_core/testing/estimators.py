from ..assessment.dataset_splitting import train_test_generator
from ..data import DataSet
from ..data.datasets import ISIMIP_dengue_harmonized


def sanity_check_estimator(estimator):
    prediction_length = 3
    dataset = ISIMIP_dengue_harmonized["vietnam"]
    train, test_generator = train_test_generator(dataset, prediction_length, n_test_sets=1)
    historic, future, _ = next(test_generator)
    predictor = estimator.train(train)
    samples = predictor.predict(historic, future)
    assert isinstance(samples, DataSet)
    for s in samples.values():
        assert len(s) == prediction_length
        assert s.samples.shape == (prediction_length, 100)

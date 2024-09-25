from chap_core.predictor.naive_predictor import MultiRegionNaivePredictor

import pytest

from chap_core.assessment.dataset_splitting import train_test_split
from chap_core.time_period import Month
from ..data_fixtures import full_data


def test_naive_predictor(full_data):
    naive_predictor = MultiRegionNaivePredictor()
    test_start = Month(2012, 7)
    train_data, test_data = train_test_split(full_data, test_start)
    naive_predictor.train(train_data)
    predictions = naive_predictor.predict(test_data)
    for loc, data in predictions.items():
        assert len(data.data()) == 1
    # assert predictions == test_data

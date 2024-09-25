from typing import Iterable
from .dataset import DataSetAdaptor

GluonTSDataSet = Iterable[dict]


class GluonTSModel:
    def __init__(self, model):
        self._model = model

    def train(self, dataset: GluonTSDataSet):
        dataset = DataSetAdaptor.to_dataset(dataset)

    def predict(self, dataset: GluonTSDataSet):
        pass

from typing import Protocol


class Predictor(Protocol):
    def __init__(self):
        pass

    def predict(self, data):
        pass

    def evaluate(self, data):
        pass

    def train(self, data):
        pass

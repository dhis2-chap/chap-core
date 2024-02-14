from typing import Protocol


class Predictor(Protocol):
    def __init__(self):
        pass

    def predict(self, x):
        pass

    def train(self, x, y):
        pass

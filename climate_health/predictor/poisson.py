from climate_health.predictor.protocol import Predictor
import sklearn.linear_model as lm


class Poisson():
    def __init__(self, alpha=1, fit_intercept=True):
        self.model = lm.PoissonRegressor(alpha=alpha, fit_intercept=fit_intercept)

    def predict(self, data):
        self.model.predict(data)

    def evaluate(self, data):
        self.model.score(data, data)

    def train(self, data):
        self.model.fit(data)

# def evaluate(predictor: Predictor, data):
#     return predictor.evaluate(data)

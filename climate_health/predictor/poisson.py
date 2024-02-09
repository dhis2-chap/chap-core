from climate_health.predictor.protocol import Predictor
import sklearn.linear_model as lm


class Poisson():
    def __init__(self, alpha=1, fit_intercept=True):
        self.model = lm.PoissonRegressor(alpha=alpha, fit_intercept=fit_intercept)

    def predict(self, X):
        """X has shape (n_samples, n_features)"""
        return self.model.predict(X)

    def train(self, data):
        """Train the model on a dataframe that has the column Disease, plus other features.
        sample_data = pd.DataFrame({
            "Disease": [1, 2, 3, 4, 5],
            "Disease1": [1, 2, 3, 4, 5],
            "Disease2": [1, 2, 3, 4, 5],
            "Rain": [1, 2, 3, 4, 5],
            "Temperature": [1, 2, 3, 4, 5],
        })
        """
        y = data.Disease
        X = data.drop(columns="Disease")
        self.model.fit(X, y)

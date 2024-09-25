import sklearn.linear_model as lm


class Poisson:
    def __init__(self, alpha=1, fit_intercept=True):
        self.model = lm.PoissonRegressor(alpha=alpha, fit_intercept=fit_intercept)

    def predict(self, x):
        """X has shape (n_samples, n_features)"""
        return self.model.predict(x)

    def train(self, x, y):
        """Train the model on a dataframe that has the column Disease, plus other features. The feature order is
         expected to be the same between training and prediction without explicit feature names.

        sample_data = pd.DataFrame({
            "Disease": [1, 2, 3, 4, 5],
            "Disease1": [1, 2, 3, 4, 5],
            "Disease2": [1, 2, 3, 4, 5],
            "Rain": [1, 2, 3, 4, 5],
            "Temperature": [1, 2, 3, 4, 5],
        })
        """
        self.model.fit(x, y)

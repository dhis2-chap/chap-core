from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PredictionsEvaluator:
    def __init__(self, predictions, ground_truths):
        self.predictions = predictions
        self.ground_truths = ground_truths
        self.metrics = self._evaluate_predictions()

    def _evaluate_predictions(self):
        metrics = {}
        metrics['mse'] = mean_squared_error(self.ground_truths, self.predictions)
        metrics['mae'] = mean_absolute_error(self.ground_truths, self.predictions)
        metrics['r2'] = r2_score(self.ground_truths, self.predictions)
        return metrics

    def get_metrics(self):
        return self.metrics

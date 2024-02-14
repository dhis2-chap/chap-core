from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import plotly.express as px

class AssessmentReport:
    def __init__(self, rmse_dict):
        self.rmse_dict = rmse_dict
        return


def make_assessment_report(prediction_dict, truth_dict) -> AssessmentReport:
    rmse_dict = {}
    for (prediction_key, prediction_value) in prediction_dict.items():
        rmse_dict[prediction_key] = root_mean_squared_error(list(truth_dict[prediction_key].values()),
                                                            list(prediction_value.values()))
    plot_rmse(rmse_dict)

    return AssessmentReport(rmse_dict)

def plot_rmse(rmse_dict):
    fig = px.line(x=list(rmse_dict.keys()),
                  y=list(rmse_dict.values()),
                  title='Root mean squared error per lag',
                  labels={'x': 'lag_ahead', 'y': 'RMSE'},
                  markers=True)
    fig.show()


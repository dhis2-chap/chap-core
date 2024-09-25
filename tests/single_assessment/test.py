from chap_core.assessment.prediction_evaluator import (
    AssessmentReport,
    make_assessment_report,
    plot_rmse,
)


def test_make_assessment_report(mocker):
    prediction_dict = {"model1": {"point1": 2, "point2": 4}}
    truth_dict = {"model1": {"point1": 1, "point2": 3}}
    expected_rmse = {"model1": 1.0}

    mocker.patch(
        "chap_core.assessment.prediction_evaluator.root_mean_squared_error",
        return_value=1.0,
    )

    mock_plot_rmse = mocker.patch(
        "chap_core.assessment.prediction_evaluator.plot_rmse"
    )

    result = make_assessment_report(prediction_dict, truth_dict, do_show=False)

    assert isinstance(
        result, AssessmentReport
    ), "The result should be an instance of AssessmentReport"
    assert (
        result.rmse_dict == expected_rmse
    ), "The RMSE dict in the result does not match the expected values"

    mock_plot_rmse.assert_called_once_with(expected_rmse)


def test_plot_rmse(mocker):
    mock_px_line = mocker.patch("plotly.express.line", autospec=True)

    rmse_dict = {"1": 0.1, "2": 0.2, "3": 0.3}

    plot_rmse(rmse_dict, do_show=False)

    mock_px_line.assert_called_once_with(
        x=list(rmse_dict.keys()),
        y=list(rmse_dict.values()),
        title="Root mean squared error per lag",
        labels={"x": "lag_ahead", "y": "RMSE"},
        markers=True,
    )

"""Tests for the report CLI command."""

from unittest.mock import MagicMock, patch

from chap_core.cli_endpoints.report import report


def test_report_cli_trains_then_reports(dumped_weekly_data_paths, tmp_path):
    _, historic_path, _ = dumped_weekly_data_paths
    out_file = tmp_path / "report.pdf"

    mock_estimator = MagicMock()
    mock_template = MagicMock()
    mock_template.__enter__.return_value = mock_template
    mock_template.__exit__.return_value = False
    mock_template.get_model.return_value = lambda: mock_estimator

    with patch(
        "chap_core.cli_endpoints.report.ModelTemplate.from_directory_or_github_url",
        return_value=mock_template,
    ) as mock_from:
        report(
            model_name=str(tmp_path / "fake_mlproject"),
            dataset_csv=str(historic_path),
            out_file=out_file,
        )

    mock_from.assert_called_once()
    mock_estimator.train.assert_called_once()
    mock_estimator.report.assert_called_once()
    # train is called before report
    train_call_index = mock_estimator.method_calls.index(
        next(c for c in mock_estimator.method_calls if c[0] == "train")
    )
    report_call_index = mock_estimator.method_calls.index(
        next(c for c in mock_estimator.method_calls if c[0] == "report")
    )
    assert train_call_index < report_call_index
    args, _ = mock_estimator.report.call_args
    assert args[1] == out_file

"""Tests for the report CLI command."""

from unittest.mock import MagicMock, patch

from chap_core.cli_endpoints.report import report


def test_report_cli_invokes_model_report(dumped_weekly_data_paths, tmp_path):
    _, historic_path, _ = dumped_weekly_data_paths
    out_file = tmp_path / "report.pdf"
    model_artifact = tmp_path / "trained_model"
    model_artifact.write_bytes(b"fake-model-bytes")

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
            model_path=tmp_path / "fake_mlproject",
            model_artifact=model_artifact,
            dataset_csv=historic_path,
            out_file=out_file,
        )

    mock_from.assert_called_once()
    mock_estimator.report.assert_called_once()
    _, kwargs = mock_estimator.report.call_args
    assert kwargs["model_artifact"] == model_artifact
    # positional args: (dataset, out_file)
    args, _ = mock_estimator.report.call_args
    assert args[1] == out_file

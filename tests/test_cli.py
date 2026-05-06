from chap_core.api import forecast
import pytest
from chap_core.util import docker_available
from chap_core.cli_endpoints.evaluate import eval_cmd
from chap_core.cli_endpoints.utils import sanity_check_model


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
@pytest.mark.skip(reason="Failing on CI")
def test_forecast_github_model():
    repo_url = "https://github.com/knutdrand/external_rmodel_example.git"
    results = forecast("external", "hydromet_5_filtered", 12, repo_url)


def test_eval_cmd(tmp_path):
    from chap_core.file_io.example_data_set import datasets
    from chap_core.api_types import BacktestParams, RunConfig

    # Export hydromet dataset to CSV for testing
    dataset = datasets["hydromet_5_filtered"].load()
    csv_path = tmp_path / "test_data.csv"
    dataset.to_csv(csv_path)

    # Prepare parameters
    backtest_params = BacktestParams(n_periods=3, n_splits=2, stride=1)
    run_config = RunConfig()

    # Run eval_cmd with CSV
    output_file = tmp_path / "evaluation.nc"
    eval_cmd(
        model_name="https://github.com/dhis2-chap/minimalist_example_lag",
        dataset_csv=csv_path,
        output_file=output_file,
        backtest_params=backtest_params,
        run_config=run_config,
    )

    # Verify output file was created
    assert output_file.exists()


def test_eval_cmd_with_data_source_mapping(tmp_path):
    import json

    import pandas as pd

    from chap_core.api_types import BacktestParams, RunConfig
    from chap_core.file_io.example_data_set import datasets

    # Export hydromet dataset to CSV for testing
    dataset = datasets["hydromet_5_filtered"].load()
    csv_path = tmp_path / "test_data.csv"
    dataset.to_csv(csv_path)

    # Rename columns to simulate a CSV with different column names
    df = pd.read_csv(csv_path)
    df.rename(columns={"rainfall": "precip", "disease_cases": "cases"}, inplace=True)
    renamed_csv_path = tmp_path / "renamed_data.csv"
    df.to_csv(renamed_csv_path, index=False)

    # Create mapping JSON file
    mapping = {"rainfall": "precip", "disease_cases": "cases"}
    mapping_path = tmp_path / "column_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)

    # Prepare parameters
    backtest_params = BacktestParams(n_periods=3, n_splits=2, stride=1)
    run_config = RunConfig()

    # Run eval_cmd with the mapping
    output_file = tmp_path / "evaluation.nc"
    eval_cmd(
        model_name="https://github.com/dhis2-chap/minimalist_example_lag",
        dataset_csv=renamed_csv_path,
        output_file=output_file,
        backtest_params=backtest_params,
        run_config=run_config,
        data_source_mapping=mapping_path,
    )

    # Verify output file was created
    assert output_file.exists()

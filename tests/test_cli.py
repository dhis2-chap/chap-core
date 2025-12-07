from chap_core.api import forecast
import pytest
from chap_core.util import docker_available
from chap_core.cli_endpoints.evaluate import evaluate_hpo, evaluate2
from chap_core.cli_endpoints.utils import sanity_check_model


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
@pytest.mark.skip(reason="Failing on CI")
def test_forecast_github_model():
    repo_url = "https://github.com/knutdrand/external_rmodel_example.git"
    results = forecast("external", "hydromet_5_filtered", 12, repo_url)


@pytest.mark.xfail(reason="Error in use_option_values: TODO: lilu")
def test_hpo_evaluate(data_path):
    hpo_config_yaml = data_path / "hpo_config.yaml"
    evaluate_hpo(
        model_name="https://github.com/dhis2-chap/chtorch",
        dataset_name="hydromet_5_filtered",
        model_configuration_yaml=hpo_config_yaml,
    )
    # chap evaluate-hpo --model_name ../../chtorch --dataset_name hydromet_5_filtered --model_configuration_yaml config1.yaml


# @pytest.mark.xfail(reason="Not implemented yet")
def test_evaluate2(tmp_path):
    from chap_core.file_io.example_data_set import datasets
    from chap_core.api_types import BackTestParams, RunConfig

    # Export hydromet dataset to CSV for testing
    dataset = datasets["hydromet_5_filtered"].load()
    csv_path = tmp_path / "test_data.csv"
    dataset.to_csv(csv_path)

    # Prepare parameters
    backtest_params = BackTestParams(n_periods=3, n_splits=2, stride=1)
    run_config = RunConfig()

    # Run evaluate2 with CSV
    output_file = tmp_path / "evaluation.nc"
    evaluate2(
        model_name="https://github.com/dhis2-chap/minimalist_example_lag",
        dataset_csv=csv_path,
        output_file=output_file,
        backtest_params=backtest_params,
        run_config=run_config,
    )

    # Verify output file was created
    assert output_file.exists()

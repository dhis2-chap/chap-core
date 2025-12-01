from chap_core.api import forecast
import pytest
from chap_core.util import docker_available
from chap_core.cli import sanity_check_model, evaluate_hpo, evaluate2


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


#@pytest.mark.xfail(reason="Not implemented yet")
def test_evaluate2():
    evaluate2(
        "https://github.com/dhis2-chap/minimalist_example_lag",
        dataset_name="hydromet_5_filtered",
        prediction_length=3,
        output_file="tmp.nc",
    )

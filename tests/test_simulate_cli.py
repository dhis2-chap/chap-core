"""Tests for the simulate CLI command."""

import pandas as pd
import pytest
import yaml

from chap_core.api_types import RunConfig
from chap_core.cli_endpoints.simulate import simulate
from chap_core.exceptions import InvalidModelException
from chap_core.models.model_template import ModelTemplate
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@pytest.fixture
def covariates_csv(data_path, tmp_path):
    path = tmp_path / "covariates.csv"
    pd.read_csv(data_path / "laos_subset.csv").drop(columns=["disease_cases"]).to_csv(path, index=False)
    return path


def test_simulate_cli_adds_simulated_target(models_path, covariates_csv, tmp_path):
    out_file = tmp_path / "simulated.csv"
    simulate(
        model_name=str(models_path / "climate_simulation_model"),
        dataset_csv=covariates_csv,
        out_file=out_file,
        run_config=RunConfig(ignore_environment=True),
    )

    covariates = pd.read_csv(covariates_csv)
    simulated = pd.read_csv(out_file)
    assert len(simulated) == len(covariates)
    assert set(simulated.location) == set(covariates.location)
    assert (simulated.disease_cases >= 0).all()


def test_simulate_cli_uses_model_configuration(models_path, covariates_csv, tmp_path):
    def run(seed):
        config_file = tmp_path / f"config_{seed}.yaml"
        config_file.write_text(yaml.dump({"user_option_values": {"seed": seed}}))
        out_file = tmp_path / f"simulated_{seed}.csv"
        simulate(
            model_name=str(models_path / "climate_simulation_model"),
            dataset_csv=covariates_csv,
            out_file=out_file,
            run_config=RunConfig(ignore_environment=True),
            model_configuration_yaml=config_file,
        )
        return pd.read_csv(out_file).disease_cases.tolist()

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_simulate_without_entry_point_raises(models_path, covariates_csv):
    template = ModelTemplate.from_directory_or_github_url(
        models_path / "naive_python_model_with_mlproject_file", ignore_env=True
    )
    model = template.get_model()
    with pytest.raises(InvalidModelException):
        model.simulate(DataSet.from_csv(covariates_csv))

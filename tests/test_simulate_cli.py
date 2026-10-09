"""Tests for the simulate CLI command."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

from chap_core.api_types import RunConfig
from chap_core.cli_endpoints.simulate import simulate
from chap_core.exceptions import InvalidModelException, ModelFailedException
from chap_core.external.model_configuration import CommandConfig, EntryPointConfig, ModelTemplateConfigV2
from chap_core.models.external_model import ExternalModel
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


def test_simulate_cli_continues_from_previous_cases(models_path, covariates_csv, tmp_path):
    def run(dataset_csv, out_file):
        simulate(
            model_name=str(models_path / "climate_simulation_model"),
            dataset_csv=dataset_csv,
            out_file=out_file,
            run_config=RunConfig(ignore_environment=True),
        )
        return pd.read_csv(out_file).sort_values(["location", "time_period"]).reset_index(drop=True)

    covariates = pd.read_csv(covariates_csv)
    first_half = tmp_path / "first_half.csv"
    covariates[covariates.time_period < "2012-07"].to_csv(first_half, index=False)
    previous = run(first_half, tmp_path / "previous.csv")

    continued_input = tmp_path / "continued_input.csv"
    new_rows = covariates[covariates.time_period >= "2012-07"]
    pd.concat([previous, new_rows]).to_csv(continued_input, index=False)
    continued = run(continued_input, tmp_path / "continued.csv")

    full = run(covariates_csv, tmp_path / "full.csv")
    old = continued.time_period < "2012-07"
    assert continued.loc[old, "disease_cases"].tolist() == previous.disease_cases.tolist()
    assert continued.disease_cases.notna().all()
    assert continued.disease_cases.tolist() == full.disease_cases.tolist()


def test_simulate_rejects_output_that_changes_observed_cases(data_path, tmp_path):
    def overwrite_cases(covariates, output_file, polygons_file_name):
        df = pd.read_csv(tmp_path / covariates, index_col=0)
        df["disease_cases"] = 0
        df.to_csv(tmp_path / output_file, index=False)

    runner = MagicMock()
    runner.simulate.side_effect = overwrite_cases
    command = CommandConfig(command="")
    model = ExternalModel(
        runner,
        name="overwriting_simulator",
        working_dir=str(tmp_path),
        model_information=ModelTemplateConfigV2(
            name="overwriting_simulator",
            entry_points=EntryPointConfig(train=command, predict=command, simulate=command),
        ),
    )
    df = pd.read_csv(data_path / "laos_subset.csv")
    df.loc[df.time_period >= "2012-07", "disease_cases"] = np.nan
    with pytest.raises(ModelFailedException, match="changed observed values"):
        model.simulate(DataSet.from_pandas(df))

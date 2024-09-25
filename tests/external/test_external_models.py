import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.datatypes import ClimateHealthTimeSeries
from chap_core.testing.external_model import sanity_check_external_model

logging.basicConfig(level=logging.INFO)
from chap_core.external.external_model import (
    run_command,
    get_model_from_directory_or_github_url,
)
from chap_core.util import conda_available, docker_available, pyenv_available


@pytest.mark.skipif(not docker_available(), reason="Requires docker")
def test_python_model_from_folder_with_mlproject_file(models_path):
    path = models_path / "naive_python_model_with_mlproject_file"
    model = get_model_from_directory_or_github_url(path)


def get_dataset_from_yaml(yaml_path: Path, datatype=ClimateHealthTimeSeries):
    specs = yaml.load(yaml_path.read_text(), Loader=yaml.FullLoader)
    if "demo_data" in specs:
        path = yaml_path.parent / specs["demo_data"]
        df = pd.read_csv(path)
    if "demo_data_adapter" in specs:
        for to_name, from_name in specs["demo_data_adapter"].items():
            if "{" in from_name:
                new_col = [
                    from_name.format(**df.iloc[i].to_dict()) for i in range(len(df))
                ]
                df[to_name] = new_col
            else:
                df[to_name] = df[from_name]
    # df['disease_cases'] = np.arange(len(df))

    return DataSet.from_pandas(df, datatype)


@pytest.mark.skipif(not conda_available(), reason="requires conda")
def test_run_conda():
    assert conda_available()
    # testing that running command with conda works
    command = "conda --version"
    run_command(command)


def test_run_command():
    command = "echo 'hi'"
    run_command(command)

    with pytest.raises(Exception):
        run_command("this_command_does_not_exist")


def test_get_model_from_github():
    repo_url = "https://github.com/knutdrand/external_rmodel_example.git"
    model = get_model_from_directory_or_github_url(repo_url)
    assert model.name == "example_model"


def test_get_model_from_local_directory(models_path):
    repo_url = models_path / "ewars_Plus"
    model = get_model_from_directory_or_github_url(repo_url)
    assert model.name == "ewars_Plus"


@pytest.mark.skipif(not pyenv_available(), reason="requires pyenv")
@pytest.mark.slow
def test_external_sanity(models_path):
    sanity_check_external_model(models_path / "naive_python_model_with_mlproject_file")


@pytest.mark.skipif(not pyenv_available(), reason="requires pyenv")
@pytest.mark.slow
@pytest.mark.skip(reason="Unstable")
def test_external_sanity_deepar(models_path):
    sanity_check_external_model(models_path / "deepar")


@pytest.mark.skipif(not docker_available(), reason="requires pyenv")
@pytest.mark.slow
def test_external_sanity_deepar(models_path):
    sanity_check_external_model("https://github.com/sandvelab/chap_auto_ewars")

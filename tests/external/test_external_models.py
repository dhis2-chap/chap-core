import logging
from pathlib import Path
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.exceptions import InvalidModelException, ModelFailedException
from chap_core.file_io.example_data_set import datasets, DataSetType
import pandas as pd
import pytest
import yaml

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.datatypes import ClimateHealthTimeSeries
from chap_core.testing.external_model import sanity_check_external_model

logging.basicConfig(level=logging.INFO)
from chap_core.external.external_model import get_model_from_directory_or_github_url
from chap_core.util import docker_available, pyenv_available


@pytest.mark.skipif(not docker_available(), reason="Requires docker")
def test_python_model_from_folder_with_mlproject_file(models_path):
    path = models_path / "naive_python_model_with_mlproject_file"
    model = get_model_from_directory_or_github_url(path)


@pytest.fixture
def dataset():
    dataset_name = "ISIMIP_dengue_harmonized"
    dataset = datasets[dataset_name]
    dataset = dataset.load()
    dataset = dataset["brazil"]
    return dataset


#@pytest.mark.skip(reason="Under development")
def test_python_model_from_folder_with_mlproject_file_that_fails(models_path, dataset):
    path = models_path / "naive_python_model_with_mlproject_file_failing"
    model = get_model_from_directory_or_github_url(path, ignore_env=True)
    with pytest.raises(ModelFailedException):
        result = evaluate_model(model, dataset)


def test_get_model_from_github():
    repo_url = "https://github.com/knutdrand/external_rmodel_example.git"
    with pytest.raises(InvalidModelException):
        model = get_model_from_directory_or_github_url(repo_url)


@pytest.mark.skip(reason="This model does not have a mlproject file, using old yml spec")
def test_get_model_from_local_directory(models_path):
    repo_url = models_path / "ewars_Plus"
    model = get_model_from_directory_or_github_url(repo_url)
    assert model.name == "ewars_Plus"


@pytest.mark.skipif(not pyenv_available(), reason="requires pyenv")
@pytest.mark.slow
def test_external_sanity(models_path):
    sanity_check_external_model(models_path / "naive_python_model_with_mlproject_file")


@pytest.mark.skipif(not docker_available(), reason="requires pyenv")
@pytest.mark.slow
def test_external_sanity_deepar(models_path):
    sanity_check_external_model("https://github.com/dhis2-chap/chap_auto_ewars")

from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.exceptions import InvalidModelException, ModelFailedException
from chap_core.file_io.example_data_set import datasets
import pytest
from chap_core.geometry import Polygons
from chap_core.testing.external_model import sanity_check_external_model
from chap_core.external.external_model import get_model_from_directory_or_github_url
from chap_core.util import docker_available, pyenv_available


@pytest.mark.skipif(not docker_available(), reason="Requires docker")
def test_python_model_from_folder_with_mlproject_file(models_path):
    path = models_path / "naive_python_model_with_mlproject_file"
    get_model_from_directory_or_github_url(path)


@pytest.fixture
def dataset():
    dataset_name = "ISIMIP_dengue_harmonized"
    dataset = datasets[dataset_name]
    dataset = dataset.load()
    dataset = dataset["brazil"]
    return dataset


def test_python_model_from_folder_with_mlproject_file_that_fails(models_path, dataset):
    path = models_path / "naive_python_model_with_mlproject_file_failing"
    model = get_model_from_directory_or_github_url(path, ignore_env=True)
    with pytest.raises(ModelFailedException):
        evaluate_model(model, dataset)


def test_get_model_from_github():
    repo_url = "https://github.com/knutdrand/external_rmodel_example.git"
    with pytest.raises(InvalidModelException):
        get_model_from_directory_or_github_url(repo_url)


@pytest.mark.skip(reason="This model does not have a mlproject file, using old yml spec")
def test_get_model_from_local_directory(models_path):
    repo_url = models_path / "ewars_Plus"
    model = get_model_from_directory_or_github_url(repo_url)
    assert model.name == "ewars_Plus"


@pytest.mark.skipif(not pyenv_available(), reason="requires pyenv")
@pytest.mark.slow
@pytest.mark.skip(reason="Failing")
def test_external_sanity(models_path):
    sanity_check_external_model(models_path / "naive_python_model_with_mlproject_file")


@pytest.mark.skipif(not docker_available(), reason="requires pyenv")
@pytest.mark.slow
def test_external_sanity_deepar(models_path):
    sanity_check_external_model("https://github.com/dhis2-chap/chap_auto_ewars")


@pytest.mark.skip(reason="Under development")
def test_that_polygons_are_sent_to_runners_through_external_model(data_path, dataset):
    # todo: implement with dataset/polygons that are compatible
    # init external model with a runner (create a dummy command line model)
    # check that commands get polygons correctly by fetching the runner and
    # inspect the commands
    polygons = Polygons.from_file(data_path / "example_polygons.geojson").data
    #dataset.set_polygons(polygons)
    

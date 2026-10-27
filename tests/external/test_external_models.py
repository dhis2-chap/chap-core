from unittest.mock import Mock

import pytest

from chap_core import get_temp_dir
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.exceptions import InvalidModelException, ModelFailedException
from chap_core.external.model_configuration import (
    CommandConfig,
    EntryPointConfig,
    ModelTemplateConfigV2,
)
from chap_core.models.external_model import ExternalModel
from chap_core.models.utils import get_model_template_from_directory_or_github_url
from chap_core.util import docker_available, uv_available


@pytest.mark.skipif(not docker_available(), reason="Requires docker")
def test_python_model_from_folder_with_mlproject_file(models_path):
    path = models_path / "naive_python_model_with_mlproject_file"
    get_model_template_from_directory_or_github_url(path)


def test_python_model_from_folder_with_mlproject_file_that_fails(models_path, dataset):
    path = models_path / "naive_python_model_with_mlproject_file_failing"
    template = get_model_template_from_directory_or_github_url(path, ignore_env=True)
    model = template.get_model()
    with pytest.raises(ModelFailedException):
        evaluate_model(model, dataset)


def test_get_model_from_github():
    repo_url = "https://github.com/knutdrand/external_rmodel_example.git"
    with pytest.raises(InvalidModelException):
        get_model_template_from_directory_or_github_url(repo_url)


@pytest.mark.skipif(not docker_available(), reason="requires pyenv")
@pytest.mark.slow
def test_external_sanity_deepar(dataset):
    folder_path = "https://github.com/dhis2-chap/minimalist_example"
    template = get_model_template_from_directory_or_github_url(folder_path, run_dir_type="latest")
    model = template.get_model()
    # model = get_model_from_directory_or_github_url(folder_path, run_dir_type="latest")
    out_file = get_temp_dir() / "deepar_evaluation_report.html"
    # make sure temp_dir exists
    out_file.parent.mkdir(parents=True, exist_ok=True)
    evaluate_model(model, dataset, report_filename=out_file)

    # sanity_check_external_model("https://github.com/dhis2-chap/minimalist_example")


@pytest.mark.skipif(not uv_available(), reason="Requires uv")
def test_uv_model_from_folder(models_path):
    path = models_path / "naive_python_model_uv"
    template = get_model_template_from_directory_or_github_url(path)
    assert template.name == "naive_python_uv"

    # dataset.set_polygons(polygons)


def _report_model_template_config(with_report: bool) -> ModelTemplateConfigV2:
    report = CommandConfig(command="python report.py {model} {historic_data} {out_file}") if with_report else None
    return ModelTemplateConfigV2(
        name="test_report_model",
        entry_points=EntryPointConfig(
            train=CommandConfig(command="python train.py {train_data} {model}"),
            predict=CommandConfig(command="python predict.py {model} {historic_data} {future_data} {out_file}"),
            report=report,
        ),
    )


def test_external_model_report_invokes_runner(weekly_full_data, tmp_path):
    working_dir = tmp_path / "workdir"
    working_dir.mkdir()
    out_file = tmp_path / "report.pdf"

    def _fake_report(model, historic, out_filename, polygons):
        # The model command runs with cwd=working_dir, so it writes the report
        # there using the relative filename it was given.
        (working_dir / out_filename).write_bytes(b"%PDF-fake")

    runner = Mock()
    runner.report.side_effect = _fake_report
    model = ExternalModel(
        runner=runner,
        name="test_report_model",
        working_dir=str(working_dir),
        model_information=_report_model_template_config(with_report=True),
    )

    model.report(weekly_full_data, out_file)

    runner.report.assert_called_once()
    args, _ = runner.report.call_args
    assert args[0] == "model"
    assert args[1] == "report_historic_data.csv"
    # The runner is given a relative filename, not the user's full out path,
    # because the model command runs in working_dir.
    assert args[2] == "report.pdf"
    assert (working_dir / "report_historic_data.csv").exists()
    # The generated file is copied from working_dir to the user-supplied path.
    assert out_file.read_bytes() == b"%PDF-fake"


def test_external_model_report_without_entry_point_raises(weekly_full_data, tmp_path):
    runner = Mock()
    model = ExternalModel(
        runner=runner,
        name="test_report_model",
        working_dir=str(tmp_path),
        model_information=_report_model_template_config(with_report=False),
    )
    with pytest.raises(InvalidModelException):
        model.report(weekly_full_data, tmp_path / "report.pdf")
    runner.report.assert_not_called()


def _model_info_template_config() -> ModelTemplateConfigV2:
    return ModelTemplateConfigV2(
        name="test_model_info_model",
        entry_points=EntryPointConfig(
            train=CommandConfig(command="python train.py {train_data} {model}"),
            predict=CommandConfig(command="python predict.py {model} {historic_data} {future_data} {out_file}"),
            model_info=CommandConfig(command="python model_info.py {out_file}"),
        ),
    )


def test_external_model_model_info_invokes_runner(tmp_path):
    working_dir = tmp_path / "workdir"
    working_dir.mkdir()

    def _fake_model_info(out_filename):
        (working_dir / out_filename).write_text('{"context_length" : 12}')

    runner = Mock()
    runner.model_info.side_effect = _fake_model_info
    model = ExternalModel(
        runner=runner,
        name="test_model_info_model",
        working_dir=str(working_dir),
        model_information=_model_info_template_config(),
    )

    model_info = model.model_info()
    assert model_info is not None

    runner.model_info.assert_called_once()
    args, _ = runner.model_info.call_args
    assert args[0] == "model_info.json"

    assert (working_dir / "model_info.json").exists()
    assert model_info.context_length == 12


def test_external_model_without_model_info_entry_point_returns_none(tmp_path):
    working_dir = tmp_path / "workdir"
    working_dir.mkdir()

    runner = Mock()
    model = ExternalModel(
        runner=runner,
        name="test_model_info_model",
        working_dir=str(working_dir),
        model_information=ModelTemplateConfigV2(
            name="test_model_info_model",
            entry_points=EntryPointConfig(
                train=CommandConfig(command="python train.py {train_data} {model}"),
                predict=CommandConfig(command="python predict.py {model} {historic_data} {future_data} {out_file}"),
            ),
        ),
    )

    assert model.model_info() == None
    runner.model_info.assert_not_called()


def test_external_model_model_info_rejects_zero(tmp_path):
    working_dir = tmp_path / "workdir"
    working_dir.mkdir()

    def _fake_model_info(out_filename):
        (working_dir / out_filename).write_text('{"context_length" : 0}')

    runner = Mock()
    runner.model_info.side_effect = _fake_model_info
    model = ExternalModel(
        runner=runner,
        name="test_model_info_model",
        working_dir=str(working_dir),
        model_information=_model_info_template_config(),
    )

    with pytest.raises(ModelFailedException):
        model.model_info()

from unittest.mock import patch

import pytest

from chap_core.exceptions import InvalidModelException
from chap_core.file_io.example_data_set import datasets
from chap_core.models.external_chapkit_model import ExternalChapkitModelTemplate
from chap_core.models.mlproject_chapkit import CHAPKIT_MLPROJECT_RUN, chapkit_service_launch, check_chapkit_version
from chap_core.models.utils import get_model_template_from_directory_or_github_url


@pytest.fixture
def orchestrator_url_set(monkeypatch):
    monkeypatch.setenv("SERVICEKIT_ORCHESTRATOR_URL", "http://chap:8000/v2/services/$register")


def test_ignore_env_runs_chapkit_directly(models_path, orchestrator_url_set):
    command, env = chapkit_service_launch(
        models_path / "naive_python_model_with_mlproject_file" / "MLproject", ignore_env=True
    )
    assert command == CHAPKIT_MLPROJECT_RUN
    assert "SERVICEKIT_ORCHESTRATOR_URL" not in env


def test_uv_env_syncs_and_runs_chapkit_through_uv(models_path):
    model_dir = models_path / "naive_python_model_uv"
    with patch("chap_core.models.mlproject_chapkit.run_command") as run_command:
        command, env = chapkit_service_launch(model_dir / "MLproject")
    assert run_command.call_args.args[0] == "uv sync"
    assert command == ["uv", "run", "--no-sync", *CHAPKIT_MLPROJECT_RUN]
    assert env["UV_PROJECT_ENVIRONMENT"] == str(model_dir.resolve() / ".venv")


def test_missing_uv_fails_before_syncing(models_path):
    with (
        patch("chap_core.models.mlproject_chapkit.shutil.which", return_value=None),
        patch("chap_core.models.mlproject_chapkit.run_command") as run_command,
    ):
        with pytest.raises(InvalidModelException, match="needs `uv` on PATH"):
            chapkit_service_launch(models_path / "naive_python_model_uv" / "MLproject")
    run_command.assert_not_called()


def test_installed_chapkit_is_recent_enough():
    check_chapkit_version()


def test_old_chapkit_is_rejected():
    with patch("chap_core.models.mlproject_chapkit.importlib.metadata.version", return_value="2.1.0"):
        with pytest.raises(InvalidModelException, match="needs chapkit >= 2.3.1, but 2.1.0 is installed"):
            check_chapkit_version()


@pytest.mark.parametrize(
    "model_dir",
    ["naive_python_model_with_mlproject_file_and_docker", "naive_python_model_with_mlproject_file"],
)
def test_unsupported_env_is_rejected(models_path, model_dir):
    with pytest.raises(InvalidModelException, match="not supported"):
        chapkit_service_launch(models_path / model_dir / "MLproject")


def test_as_chapkit_returns_directory_mode_template(models_path, tmp_path):
    template = get_model_template_from_directory_or_github_url(
        str(models_path / "naive_python_model_with_mlproject_file"),
        base_working_dir=tmp_path,
        ignore_env=True,
        as_chapkit=True,
    )
    assert isinstance(template, ExternalChapkitModelTemplate)
    assert not template.is_url_mode


def test_as_chapkit_rejects_service_url():
    with pytest.raises(ValueError, match="as-chapkit"):
        get_model_template_from_directory_or_github_url("http://localhost:8000", as_chapkit=True)


def test_as_chapkit_rejects_is_chapkit_model(models_path):
    with pytest.raises(ValueError, match="as-chapkit"):
        get_model_template_from_directory_or_github_url(
            str(models_path / "naive_python_model_uv"), is_chapkit_model=True, as_chapkit=True
        )


def test_as_chapkit_rejects_dry_run(models_path, tmp_path):
    with pytest.raises(ValueError, match="dry-run"):
        get_model_template_from_directory_or_github_url(
            str(models_path / "naive_python_model_with_mlproject_file"),
            base_working_dir=tmp_path,
            ignore_env=True,
            as_chapkit=True,
            dry_run=True,
        )
    assert not any(tmp_path.iterdir())


@pytest.mark.slow
def test_as_chapkit_starts_trains_and_stops_service(models_path, tmp_path):
    dataset = datasets["ISIMIP_dengue_harmonized"].load()["vietnam"]
    template = get_model_template_from_directory_or_github_url(
        str(models_path / "naive_python_model_with_mlproject_file"),
        base_working_dir=tmp_path,
        ignore_env=True,
        as_chapkit=True,
    )
    assert isinstance(template, ExternalChapkitModelTemplate)
    with template:
        model = template.get_model({}, prediction_length=3)
        model.train(dataset)
        assert template.is_healthy()
    assert template.client is None

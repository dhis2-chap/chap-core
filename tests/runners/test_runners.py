from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

from chap_core.exceptions import CommandLineException
from chap_core.runners.command_line_runner import CommandLineRunner
from chap_core.runners.docker_runner import DockerRunner
from chap_core.runners.uv_runner import UvRunner, UvTrainPredictRunner
from chap_core.runners.renv_runner import RenvRunner, RenvTrainPredictRunner
from chap_core.runners.conda_runner import CondaRunner, CondaTrainPredictRunner
from chap_core.runners.helper_functions import get_train_predict_runner_from_model_template_config
from chap_core.external.model_configuration import (
    ModelTemplateConfigV2,
    EntryPointConfig,
    CommandConfig,
)
import pytest

from chap_core.util import docker_available


def test_command_line_runner():
    command = "echo 'test'"
    runner = CommandLineRunner(Path("."))
    runner.run_command("")


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
def test_docker_runner():
    docker_image = "ubuntu:noble"
    runner = DockerRunner(docker_image, Path("."))
    output = runner.run_command("echo 'test'")
    assert "test" in output


def test_run_command():
    command = "echo 'test'"
    output = CommandLineRunner("./").run_command(command)
    assert "test" in output, "Output from command not as expected, output is: " + output

    # also test that stderr is captured
    command = "echo 'test2' >&2"
    output = CommandLineRunner("./").run_command(command)
    assert "test2" in output, "Output from command not as expected, output is: " + output


def test_uv_runner_prepends_uv_run():
    """Test that UvRunner correctly prepends 'uv run' to commands"""
    with patch("chap_core.runners.uv_runner.run_command") as mock_run_command:
        mock_run_command.return_value = "test output"
        runner = UvRunner(Path("."))
        runner.run_command("python main.py train data.csv model.pkl")
        mock_run_command.assert_called_once_with("uv run python main.py train data.csv model.pkl", Path("."), env=ANY)


def test_uv_train_predict_runner_formats_commands():
    """Test that UvTrainPredictRunner formats train and predict commands correctly"""
    with patch("chap_core.runners.uv_runner.run_command") as mock_run_command:
        mock_run_command.return_value = "test output"
        runner = UvTrainPredictRunner(
            UvRunner(Path(".")),
            train_command="python main.py train {train_data} {model}",
            predict_command="python main.py predict {model} {historic_data} {future_data} {out_file}",
        )
        runner.train("train.csv", "model.pkl")
        mock_run_command.assert_called_with("uv run python main.py train train.csv model.pkl", Path("."), env=ANY)


def test_runner_selection_with_uv_env(tmp_path):
    """Test that get_train_predict_runner_from_model_template_config returns UvTrainPredictRunner when uv_env is set"""
    config = ModelTemplateConfigV2(
        name="test_model",
        uv_env="pyproject.toml",
        entry_points=EntryPointConfig(
            train=CommandConfig(
                command="python main.py train {train_data} {model}",
                parameters={"train_data": "str", "model": "str"},
            ),
            predict=CommandConfig(
                command="python main.py predict {model} {historic_data} {future_data} {out_file}",
                parameters={
                    "model": "str",
                    "historic_data": "str",
                    "future_data": "str",
                    "out_file": "str",
                },
            ),
        ),
    )
    runner = get_train_predict_runner_from_model_template_config(config, tmp_path)
    assert isinstance(runner, UvTrainPredictRunner)


def test_renv_runner_restores_and_runs_command(tmp_path):
    """Test that RenvRunner runs renv::restore() before first command"""
    (tmp_path / "renv.lock").write_text("{}")

    with patch("chap_core.runners.renv_runner.run_command") as mock_run_command:
        mock_run_command.return_value = "test output"
        runner = RenvRunner(tmp_path)
        runner.run_command("Rscript main.R train data.csv model.rds")

        assert mock_run_command.call_count == 2
        calls = mock_run_command.call_args_list
        assert "renv::restore" in calls[0][0][0]
        assert "Rscript main.R train data.csv model.rds" in calls[1][0][0]


def test_renv_runner_only_restores_once(tmp_path):
    """Test that RenvRunner only runs restore once across multiple commands"""
    (tmp_path / "renv.lock").write_text("{}")

    with patch("chap_core.runners.renv_runner.run_command") as mock_run_command:
        mock_run_command.return_value = "test output"
        runner = RenvRunner(tmp_path)
        runner.run_command("Rscript main.R train data.csv model.rds")
        runner.run_command("Rscript main.R predict model.rds h.csv f.csv out.csv")

        # 1 restore + 2 commands = 3 calls
        assert mock_run_command.call_count == 3


def test_renv_runner_fails_without_lockfile(tmp_path):
    """Test that RenvRunner raises error if renv.lock is missing"""
    runner = RenvRunner(tmp_path)
    with pytest.raises(CommandLineException, match="renv.lock not found"):
        runner.run_command("Rscript main.R train data.csv model.rds")


def test_renv_runner_skips_restore_when_disabled(tmp_path):
    """Test that RenvRunner skips restore when auto_restore=False"""
    (tmp_path / "renv.lock").write_text("{}")

    with patch("chap_core.runners.renv_runner.run_command") as mock_run_command:
        mock_run_command.return_value = "test output"
        runner = RenvRunner(tmp_path, auto_restore=False)
        runner.run_command("Rscript main.R train data.csv model.rds")

        # Only 1 call (no restore)
        assert mock_run_command.call_count == 1


def test_renv_train_predict_runner_formats_commands(tmp_path):
    """Test that RenvTrainPredictRunner formats train and predict commands correctly"""
    (tmp_path / "renv.lock").write_text("{}")

    with patch("chap_core.runners.renv_runner.run_command") as mock_run_command:
        mock_run_command.return_value = "test output"
        runner = RenvTrainPredictRunner(
            RenvRunner(tmp_path),
            train_command="Rscript main.R train {train_data} {model}",
            predict_command="Rscript main.R predict {model} {historic_data} {future_data} {out_file}",
        )
        runner.train("train.csv", "model.rds")

        # Last call should be the formatted train command
        last_call = mock_run_command.call_args_list[-1]
        assert "Rscript main.R train train.csv model.rds" in last_call[0][0]


def test_runner_selection_with_renv_env(tmp_path):
    """Test that get_train_predict_runner_from_model_template_config returns RenvTrainPredictRunner when renv_env is set"""
    config = ModelTemplateConfigV2(
        name="test_r_model",
        renv_env="renv.lock",
        entry_points=EntryPointConfig(
            train=CommandConfig(
                command="Rscript main.R train {train_data} {model}",
                parameters={"train_data": "str", "model": "str"},
            ),
            predict=CommandConfig(
                command="Rscript main.R predict {model} {historic_data} {future_data} {out_file}",
                parameters={
                    "model": "str",
                    "historic_data": "str",
                    "future_data": "str",
                    "out_file": "str",
                },
            ),
        ),
    )
    runner = get_train_predict_runner_from_model_template_config(config, tmp_path)
    assert isinstance(runner, RenvTrainPredictRunner)


def test_conda_runner_prepends_conda_run(tmp_path):
    """Test that CondaRunner correctly formats commands with conda run"""
    (tmp_path / "environment.yaml").write_text("name: test\ndependencies:\n  - python")

    with patch("chap_core.runners.conda_runner.run_command") as mock_run_command:
        mock_run_command.return_value = "test output"
        runner = CondaRunner(tmp_path, "environment.yaml")
        runner.run_command("python main.py train data.csv model.pkl")

        assert mock_run_command.call_count == 2
        # First call creates the environment
        create_call = mock_run_command.call_args_list[0][0][0]
        assert "conda env create" in create_call
        assert "-f environment.yaml" in create_call
        # Second call runs the command
        run_call = mock_run_command.call_args_list[1][0][0]
        assert "conda run" in run_call
        assert "python main.py train data.csv model.pkl" in run_call


def test_conda_runner_updates_existing_env(tmp_path):
    """Test that CondaRunner uses 'conda env update' when env directory exists"""
    (tmp_path / "environment.yaml").write_text("name: test\ndependencies:\n  - python")
    (tmp_path / ".conda_env").mkdir()

    with patch("chap_core.runners.conda_runner.run_command") as mock_run_command:
        mock_run_command.return_value = "test output"
        runner = CondaRunner(tmp_path, "environment.yaml")
        runner.run_command("python main.py train data.csv model.pkl")

        create_call = mock_run_command.call_args_list[0][0][0]
        assert "conda env update" in create_call


def test_conda_runner_only_creates_env_once(tmp_path):
    """Test that CondaRunner only creates the environment once across multiple commands"""
    (tmp_path / "environment.yaml").write_text("name: test\ndependencies:\n  - python")

    with patch("chap_core.runners.conda_runner.run_command") as mock_run_command:
        mock_run_command.return_value = "test output"
        runner = CondaRunner(tmp_path, "environment.yaml")
        runner.run_command("python main.py train data.csv model.pkl")
        runner.run_command("python main.py predict model.pkl h.csv f.csv out.csv")

        # 1 env create + 2 commands = 3 calls
        assert mock_run_command.call_count == 3


def test_conda_runner_fails_without_env_file(tmp_path):
    """Test that CondaRunner raises error if environment file is missing"""
    runner = CondaRunner(tmp_path, "environment.yaml")
    with pytest.raises(FileNotFoundError, match="environment.yaml"):
        runner.run_command("python main.py train data.csv model.pkl")


def test_runner_selection_with_conda_env(tmp_path):
    """Test that get_train_predict_runner_from_model_template_config returns CondaTrainPredictRunner when conda_env is set"""
    config = ModelTemplateConfigV2(
        name="test_conda_model",
        conda_env="environment.yaml",
        entry_points=EntryPointConfig(
            train=CommandConfig(
                command="python main.py train {train_data} {model}",
                parameters={"train_data": "str", "model": "str"},
            ),
            predict=CommandConfig(
                command="python main.py predict {model} {historic_data} {future_data} {out_file}",
                parameters={
                    "model": "str",
                    "historic_data": "str",
                    "future_data": "str",
                    "out_file": "str",
                },
            ),
        ),
    )
    runner = get_train_predict_runner_from_model_template_config(config, tmp_path)
    assert isinstance(runner, CondaTrainPredictRunner)

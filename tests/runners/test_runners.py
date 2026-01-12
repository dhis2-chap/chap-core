from pathlib import Path
from unittest.mock import patch, MagicMock

from chap_core.runners.command_line_runner import CommandLineRunner
from chap_core.runners.docker_runner import DockerRunner
from chap_core.runners.uv_runner import UvRunner, UvTrainPredictRunner
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
        mock_run_command.assert_called_once_with("uv run python main.py train data.csv model.pkl", Path("."))


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
        mock_run_command.assert_called_with("uv run python main.py train train.csv model.pkl", Path("."))


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

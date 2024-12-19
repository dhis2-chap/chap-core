from pathlib import Path

from chap_core.runners.command_line_runner import CommandLineRunner
from chap_core.runners.docker_runner import DockerImageRunner, DockerRunner
import pytest

from chap_core.util import docker_available


def test_command_line_runner():
    command = "echo 'test'"
    runner = CommandLineRunner(Path("."))
    runner.run_command("")


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
def test_docker_image_runner(data_path):
    docker_image_path = "docker_example_image"
    # docker_image_path = "../../external_models/docker_r_base/"
    print(docker_image_path)
    testcommand = 'R -e \'print("test1"); print("test2")\''
    wd = data_path.parent / "tests/runners/"
    runner = DockerImageRunner(wd / docker_image_path, wd)
    output = runner.run_command(testcommand)
    assert "test1" in output


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

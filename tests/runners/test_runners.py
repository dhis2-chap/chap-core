from pathlib import Path

from chap_core.runners.command_line_runner import CommandLineRunner
from chap_core.runners.docker_runner import DockerRunner
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

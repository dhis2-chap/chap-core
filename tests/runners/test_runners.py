from pathlib import Path

from chap_core.runners.command_line_runner import CommandLineRunner
from chap_core.runners.docker_runner import DockerImageRunner, DockerRunner


def test_command_line_runner():
    command = "echo 'test'"
    runner = CommandLineRunner(Path("."))
    runner.run_command("")


def test_docker_image_runner(data_path):
    docker_image_path = "docker_example_image"
    # docker_image_path = "../../external_models/docker_r_base/"
    print(docker_image_path)
    testcommand = 'R -e \'print("test1"); print("test2")\''
    wd = data_path.parent / "tests/runners/"
    runner = DockerImageRunner(wd / docker_image_path, wd)
    output = runner.run_command(testcommand)
    assert "test1" in output


def test_docker_runner():
    docker_image = "ubuntu:noble"
    runner = DockerRunner(docker_image, Path("."))
    output = runner.run_command("echo 'test'")
    assert "test" in output

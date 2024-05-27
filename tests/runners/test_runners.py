from climate_health.runners.command_line_runner import CommandLineRunner
from climate_health.runners.docker_runner import DockerRunner


def test_command_line_runner():
    command = "echo 'test'"
    runner = CommandLineRunner()
    runner.run_command()


def test_docker_runner(data_path):
    docker_image_path = "docker_example_image"
    #docker_image_path = "../../external_models/docker_r_base/"
    testcommand = "R -e 'print(\"test1\"); print(\"test2\")'"
    runner = DockerRunner(docker_image_path, data_path.parent / "tests/runners/")
    output = runner.run_command(testcommand)
    assert "test1" in output


if __name__ == "__main__":
    test_docker_runner()

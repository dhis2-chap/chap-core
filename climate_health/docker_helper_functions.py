import logging
import os
from pathlib import Path
import docker


def create_docker_image(dockerfile_directory: str, working_dir: str="./"):
    """Creates a docker image based on path to a directory that should contain a Dockerfile.
    Uses the final directory name as the name for the image (e.g. /path/to/name/ -> name)
    working_dir: str
        Optional. The working dir that the path will be relative to.
    Returns the name.
    """
    client = docker.from_env()
    name = Path(dockerfile_directory).stem
    logging.info(f"Creating docker image {name} from Dockerfile in {dockerfile_directory}")
    dockerfile = Path(working_dir) / Path(dockerfile_directory) / "Dockerfile"
    logging.info(f"Looking for dockerfile {dockerfile}")
    response = client.api.build(fileobj=open(dockerfile, "rb"),
                                tag=name, decode=True)
    for line in response:
        if "stream" in line:
            print(line["stream"])  # .encode("utf-8"))
        else:
            print(line)

    return name


def run_command_through_docker_container(docker_image_name: str, working_directory: str, command: str):
    client = docker.from_env()
    working_dir_full_path = os.path.abspath(working_directory)
    container = client.containers.run(docker_image_name,
                                      command=command,
                                      volumes=[f"{working_dir_full_path}:/home/run/"],
                                      working_dir="/home/run",
                                      auto_remove=False,
                                      detach=True)
    output = container.attach(stdout=True, stream=False, logs=True)
    print(output)
    full_output = output
    #full_output = ""
    #for line in output:
    #    print("Line output: ", line)
    #    full_output += line.decode("utf-8")

    result = container.wait()
    exit_code = result["StatusCode"]
    assert exit_code == 0, f"Command failed with exit code {exit_code}"
    container.remove()

    return full_output

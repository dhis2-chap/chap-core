import logging
import os
from pathlib import Path

import docker

from chap_core.exceptions import DockerUnavailableError

logger = logging.getLogger(__name__)


def _docker_client_or_raise(image_name: str | None = None):
    """Return a docker client, raising DockerUnavailableError with actionable
    guidance when the daemon is not reachable.

    The message differs depending on whether chap-core is running inside the
    server container (IS_IN_DOCKER=1) or locally (CLI / dev).
    """
    try:
        return docker.from_env()
    except docker.errors.DockerException as e:
        ref = f" '{image_name}'" if image_name else ""
        if os.environ.get("IS_IN_DOCKER"):
            msg = (
                f"Cannot reach the Docker daemon from the chap-core worker. "
                f"Models with docker_env (image{ref}) cannot run on the server "
                f"because the worker has no Docker socket access. "
                f"Re-register the model without docker_env (use uv_env / renv_env / "
                f"conda_env / python_env), or run it as a chapkit sidecar."
            )
        else:
            msg = (
                f"Cannot reach the Docker daemon. The model{ref} requires Docker "
                f"to run. Start the daemon and retry: open Docker Desktop on macOS/Windows, "
                f"or run `sudo systemctl start docker` on Linux."
            )
        raise DockerUnavailableError(msg) from e


def create_docker_image(dockerfile_directory: Path | str):
    """Creates a docker image based on path to a directory that should contain a Dockerfile.
    Uses the final directory name as the name for the image (e.g. /path/to/name/ -> name)
    Returns the name.
    """
    name = Path(dockerfile_directory).stem
    logging.info(f"Creating docker image {name} from Dockerfile in {dockerfile_directory}")
    dockerfile = Path(dockerfile_directory) / "Dockerfile"
    logging.info(f"Looking for dockerfile {dockerfile}")
    with open(dockerfile, "rb") as fileobject:
        return docker_image_from_fo(fileobject, name)


def docker_image_from_fo(fileobject, name):
    client = _docker_client_or_raise(name)
    response = client.api.build(fileobj=fileobject, tag=name, decode=True)
    for _ in response:
        pass
    return name


def run_command_through_docker_container(
    docker_image_name: str, working_directory: str, command: str, remove_after_run: bool = False
):
    client = _docker_client_or_raise(docker_image_name)
    try:
        working_dir_full_path = os.path.abspath(working_directory)
    except FileNotFoundError:
        logging.error(f"Could not find working dir {working_directory}.")
        logging.error(f"Current directory is {os.getcwd()}")
        raise

    logger.debug(
        f"Running command {command} in docker image {docker_image_name} with mount {working_dir_full_path}:/home/run/"
    )
    logger.debug(
        f"Equivalent docker command: docker run -w /home/run -v {working_dir_full_path}:/home/run/ {docker_image_name} {command}"
    )
    container = client.containers.run(
        docker_image_name,
        command=command,
        volumes=[f"{working_dir_full_path}:/home/run/"],
        working_dir="/home/run",
        auto_remove=remove_after_run,
        detach=True,
    )

    result = container.wait()
    exit_code = result["StatusCode"]
    log_output = container.logs().decode("utf-8")
    assert exit_code == 0, f"Command failed with exit code {exit_code}: {log_output}"
    container.remove()

    return log_output

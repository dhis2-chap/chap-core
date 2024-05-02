from shutil import which


def conda_available():
    return which("conda") is not None


def docker_available():
    return which("docker") is not None

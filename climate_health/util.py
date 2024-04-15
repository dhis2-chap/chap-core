from shutil import which


def conda_available():
    return which("conda") is not None

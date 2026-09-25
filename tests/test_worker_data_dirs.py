"""The worker runs as a non-root user, so its image must own every volume mount point.

A fresh named volume copies the ownership of the image directory it is mounted over. If the
image has no such directory, Docker creates the volume owned by root and the worker cannot write.
"""

import pathlib
import re

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _worker(compose_file):
    return yaml.safe_load((REPO_ROOT / compose_file).read_text())["services"]["worker"]


def _run_instructions(dockerfile):
    text = (REPO_ROOT / dockerfile).read_text().replace("\\\n", " ")
    return re.findall(r"^RUN\s(.*)$", text, re.MULTILINE)


def _volume_targets():
    return [v["target"] for v in _worker("compose.yml")["volumes"] if v["type"] == "volume"]


@pytest.mark.parametrize("compose_file", ["compose.yml", "compose.dev.yml"])
def test_worker_image_owns_volume_mount_points(compose_file):
    dockerfile = _worker(compose_file)["build"]["dockerfile"]
    chowned = " ".join(run for run in _run_instructions(dockerfile) if "chown chap:chap" in run).split()
    missing = [target for target in _volume_targets() if target not in chowned]
    assert not missing, f"{dockerfile} does not create and chown {missing} for the chap user"

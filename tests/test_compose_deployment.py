"""Every deployed service must restart unattended and report health.

Health is checked as reachable at runtime, not declared in compose: chap and worker get
theirs from the image, so a compose-only assertion would miss a worker shipped with none.
"""

import pathlib
import re

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Test and integration compose files are excluded: they overlay compose.yml rather than
# declaring their own services, so restart policies reach them by inheritance.
DEPLOYMENT_COMPOSE_FILES = ["compose.yml", "compose.ghcr.yml", "compose.ewars.yml"]

# Services that overlay a compose.yml service but run to completion instead of staying up.
# They inherit its restart policy, which would restart them after they exit and mask the
# exit code the shell scripts in tests/ read back.
ONE_SHOT_OVERLAY_SERVICES = [("compose.test.yml", "chap")]

IMAGE_TO_DOCKERFILE = {
    "ghcr.io/dhis2-chap/chap-core": "Dockerfile",
    "ghcr.io/dhis2-chap/chap-worker": "Dockerfile.worker",
}


def _services(compose_file):
    content = yaml.safe_load((REPO_ROOT / compose_file).read_text())
    return (content or {}).get("services", {})


def _dockerfile_has_healthcheck(dockerfile):
    path = REPO_ROOT / dockerfile
    if not path.exists():
        return False
    return re.search(r"^HEALTHCHECK\s", path.read_text(), re.MULTILINE) is not None


def _healthcheck_source(service):
    """Which file supplies this service's healthcheck, or None if nothing does."""
    if service.get("healthcheck"):
        return "compose"

    build = service.get("build")
    if isinstance(build, dict):
        dockerfile = build.get("dockerfile", "Dockerfile")
    elif isinstance(build, str):
        dockerfile = "Dockerfile"
    else:
        dockerfile = None
    if dockerfile and _dockerfile_has_healthcheck(dockerfile):
        return dockerfile

    image = service.get("image")
    if image:
        repo = image.rsplit(":", 1)[0]
        dockerfile = IMAGE_TO_DOCKERFILE.get(repo)
        if dockerfile and _dockerfile_has_healthcheck(dockerfile):
            return dockerfile

    return None


def _all_services():
    return [(f, name, svc) for f in DEPLOYMENT_COMPOSE_FILES for name, svc in _services(f).items()]


@pytest.mark.parametrize(
    ("compose_file", "service_name", "service"),
    _all_services(),
    ids=[f"{f}:{name}" for f, name, _ in _all_services()],
)
def test_deployed_service_restarts_unattended(compose_file, service_name, service):
    assert service.get("restart") == "unless-stopped", (
        f"{compose_file}: service '{service_name}' has no restart policy, so it will not come back after a host reboot"
    )


@pytest.mark.parametrize(
    ("compose_file", "service_name", "service"),
    _all_services(),
    ids=[f"{f}:{name}" for f, name, _ in _all_services()],
)
def test_deployed_service_reports_health(compose_file, service_name, service):
    if _healthcheck_source(service) is not None:
        return

    # An image we do not build is outside this repo's control; anything we build is a gap.
    image = service.get("image", "")
    if not service.get("build") and not any(image.startswith(repo) for repo in IMAGE_TO_DOCKERFILE):
        pytest.skip(f"'{service_name}' uses external image {image!r} and declares no healthcheck")

    pytest.fail(
        f"{compose_file}: service '{service_name}' has no healthcheck in compose and its "
        f"image carries none either, so depends_on conditions and restart-on-unhealthy "
        f"cannot work for it"
    )


@pytest.mark.parametrize(("compose_file", "service_name"), ONE_SHOT_OVERLAY_SERVICES)
def test_one_shot_overlay_service_cancels_inherited_restart(compose_file, service_name):
    service = _services(compose_file)[service_name]
    assert service.get("restart") == "no", (
        f"{compose_file}: service '{service_name}' runs to completion but does not set "
        f'restart: "no", so it inherits the deployment restart policy and will be restarted '
        f"after it exits, hiding its exit code from the test scripts"
    )

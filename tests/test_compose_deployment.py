"""Every deployed service must restart unattended and report health.

Health is checked as reachable at runtime, not declared in compose: chap and worker get
theirs from the image, so a compose-only assertion would miss a worker shipped with none.
"""

import pathlib
import re

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Test and integration compose files are excluded: a restart policy there would hang CI.
DEPLOYMENT_COMPOSE_FILES = ["compose.yml", "compose.ghcr.yml", "compose.ewars.yml"]

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


def _image_repository(image):
    """Image reference without its tag, which may itself be a ${VAR:-default}."""
    return re.sub(r":[^/]*$", "", image)


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
        repo = _image_repository(image)
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


# The file users download on its own and run without a checkout.
STANDALONE_COMPOSE_FILE = "compose.ghcr.yml"

INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:?-[^}]*)?\}")


def test_standalone_compose_needs_no_env_file():
    text = (REPO_ROOT / STANDALONE_COMPOSE_FILE).read_text()
    missing_default = sorted({m.group(1) for m in INTERPOLATION.finditer(text) if not m.group(2)})
    assert not missing_default, (
        f"{STANDALONE_COMPOSE_FILE}: {missing_default} have no default, so the file cannot be "
        f"downloaded on its own and started with `docker compose up`"
    )


@pytest.mark.parametrize("service_name", ["chap", "worker"])
def test_standalone_compose_declares_own_healthcheck(service_name):
    """A pinned tag may predate the healthcheck baked into the current Dockerfiles."""
    service = _services(STANDALONE_COMPOSE_FILE)[service_name]
    assert _healthcheck_source(service) == "compose", (
        f"{STANDALONE_COMPOSE_FILE}: service '{service_name}' inherits its healthcheck from the "
        f"image, so pinning a tag published before that healthcheck existed leaves it unmonitored"
    )


@pytest.mark.parametrize("service_name", ["chap", "worker"])
def test_standalone_compose_image_tag_is_pinnable(service_name):
    image = _services(STANDALONE_COMPOSE_FILE)[service_name]["image"]
    repo = _image_repository(image)
    tag = image[len(repo) + 1 :]
    assert repo in IMAGE_TO_DOCKERFILE, f"unexpected image repository {repo!r}"
    assert tag == "${CHAP_IMAGE_TAG:-latest}", (
        f"{STANDALONE_COMPOSE_FILE}: service '{service_name}' pins {image!r}, so a specific "
        f"release cannot be deployed without editing the file"
    )


MODEL_OVERLAY_FILES = ["compose.ewars.yml"]


@pytest.mark.parametrize("compose_file", MODEL_OVERLAY_FILES)
def test_model_overlay_pins_image_by_commit(compose_file):
    """Model overlays pin a sha-<commit> build with an overridable tag variable, never latest."""
    for service_name, service in _services(compose_file).items():
        image = service["image"]
        tag = image[len(_image_repository(image)) + 1 :]
        assert re.fullmatch(r"\$\{[A-Z_]+_IMAGE_TAG:-sha-[0-9a-f]{7}\}", tag), (
            f"{compose_file}: service '{service_name}' uses {image!r}; expected "
            "${<SERVICE>_IMAGE_TAG:-sha-<commit>} so the deployment is reproducible and pinnable"
        )
        assert service.get("pull_policy") != "always", (
            f"{compose_file}: service '{service_name}' pulls on every up, which defeats the pin"
        )


def test_env_example_leaves_redis_password_unset():
    """The compose valkey service takes no --requirepass, so an AUTH from the client fails."""
    requirepass = [f for f in DEPLOYMENT_COMPOSE_FILES if "requirepass" in str(_services(f).get("redis", {}))]
    assert not requirepass, f"{requirepass} configure a redis password; update this test"

    active = [
        line
        for line in (REPO_ROOT / ".env.example").read_text().splitlines()
        if line.strip().startswith("REDIS_PASSWORD=") and line.split("=", 1)[1].strip()
    ]
    assert not active, (
        f".env.example sets {active}, so copying it to .env makes the redis client send AUTH "
        f"to a valkey service that has no password configured"
    )


BASE_COMPOSE_FILES = ["compose.yml", "compose.ghcr.yml"]

# The broker and database live here. Model services attach only to the implicit default
# network, which reaches chap and worker but not this one.
BACKEND_NETWORK = "backend"


@pytest.mark.parametrize("compose_file", BASE_COMPOSE_FILES)
def test_base_compose_keeps_broker_and_database_off_default_network(compose_file):
    content = yaml.safe_load((REPO_ROOT / compose_file).read_text())
    assert BACKEND_NETWORK in (content.get("networks") or {}), (
        f"{compose_file} does not declare the '{BACKEND_NETWORK}' network"
    )
    services = content["services"]
    for name in ("redis", "postgres"):
        assert services[name].get("networks") == [BACKEND_NETWORK], (
            f"{compose_file}: '{name}' must sit on '{BACKEND_NETWORK}' only, or any model service on the "
            f"default network can reach it"
        )
    for name in ("chap", "worker"):
        assert services[name].get("networks") == ["default", BACKEND_NETWORK], (
            f"{compose_file}: '{name}' must join both networks to serve model services and reach the broker"
        )


@pytest.mark.parametrize("compose_file", MODEL_OVERLAY_FILES + ["compose.override.yml.example"])
def test_model_services_stay_on_default_network(compose_file):
    """A model service that declares no networks is isolated automatically; opting into backend defeats that."""
    for service_name, service in _services(compose_file).items():
        assert BACKEND_NETWORK not in (service.get("networks") or []), (
            f"{compose_file}: service '{service_name}' joins '{BACKEND_NETWORK}', so it shares a network "
            f"with the Celery broker and can enqueue tasks the worker executes"
        )

import logging
import subprocess

import pytest
import yaml

from chap_core.cli import app
from chap_core.cli_endpoints.marketplace import install, uninstall, update
from chap_core.services.model_marketplace import resolve_model


def test_resolves_stable_not_latest(marketplace_model, marketplace_http):
    marketplace_model["channels"]["latest"] = "99.0.0"
    marketplace_model["versions"].insert(
        0, {**marketplace_model["versions"][0], "version": "99.0.0", "status": "unstable"}
    )
    pin = resolve_model(marketplace_model["id"])
    assert pin.version == "0.1.0"
    assert pin.image == f"{marketplace_model['source']['image']}:sha-57eeb78"
    assert len(marketplace_http) == 2


@pytest.mark.parametrize("status", ["unstable", "deprecated", "yanked"])
def test_refuses_unverified_stable(marketplace_model, marketplace_http, status):
    marketplace_model["versions"][0]["status"] = status
    with pytest.raises(ValueError, match="no verified stable"):
        resolve_model(marketplace_model["id"])


def test_refuses_missing_stable(marketplace_model, marketplace_http):
    marketplace_model["channels"]["stable"] = "missing"
    with pytest.raises(ValueError, match="no verified stable"):
        resolve_model(marketplace_model["id"])


@pytest.mark.parametrize("tag", ["latest", "sha-0000000", "sha-"])
def test_refuses_invalid_pin(marketplace_model, marketplace_http, tag):
    marketplace_model["versions"][0]["image_tag"] = tag
    with pytest.raises(ValueError, match="invalid stable image pin"):
        resolve_model(marketplace_model["id"])


def test_refuses_template(marketplace_model, marketplace_http):
    marketplace_model["kind"] = "template"
    with pytest.raises(ValueError, match="template"):
        resolve_model(marketplace_model["id"])


def test_custom_registry_url_is_used(marketplace_model, marketplace_http, monkeypatch):
    monkeypatch.setenv("CHAP_MARKETPLACE_URL", "https://models.example.org/registry/")
    resolve_model(marketplace_model["id"])
    assert marketplace_http == [
        "https://models.example.org/registry/registry.yaml",
        f"https://models.example.org/registry/models/{marketplace_model['id']}.yaml",
    ]


def test_custom_registry_requires_risk_acceptance(marketplace_model, marketplace_http, model_deployment, monkeypatch):
    monkeypatch.setenv("CHAP_MARKETPLACE_URL", "https://models.example.org/registry")
    with pytest.raises(SystemExit):
        install(marketplace_model["id"])
    model_deployment.runner.assert_not_called()
    install(marketplace_model["id"], accept_risk=True)
    config = yaml.safe_load(model_deployment.overlay.read_text())
    assert config["services"][f"marketplace-{marketplace_model['service_id']}"]["x-chap-custom"] is False


def test_accepts_full_length_sha_tag(marketplace_model, marketplace_http):
    commit = marketplace_model["versions"][0]["commit"]
    marketplace_model["versions"][0]["image_tag"] = f"sha-{commit}"
    assert resolve_model(marketplace_model["id"]).image.endswith(f":sha-{commit}")


def test_unknown_model_never_fetches_arbitrary_path(marketplace_http):
    with pytest.raises(ValueError, match="not listed"):
        resolve_model("../../custom")
    assert len(marketplace_http) == 1


def test_install_and_update_preserve_settings_and_data(marketplace_model, marketplace_http, model_deployment):
    model = marketplace_model["id"]
    app(["install", model], result_action="return_value")
    installed = yaml.safe_load(model_deployment.overlay.read_text())
    service_name = f"marketplace-{marketplace_model['service_id']}"
    service = installed["services"][service_name]
    assert service["image"].endswith(":sha-57eeb78")
    assert service["environment"]["SERVICEKIT_ORCHESTRATOR_URL"].endswith("/$$register")
    assert service["environment"]["SERVICEKIT_HOST"] == service_name
    # The database must live on the mounted volume, not the read-only root filesystem.
    assert service["environment"]["DATABASE_URL"] == "sqlite+aiosqlite:////app/data/chapkit.db"
    assert service["volumes"][0] == f"{service_name}-data:/app/data"
    assert "ports" not in service
    service["cpus"] = 2
    model_deployment.overlay.write_text(yaml.safe_dump(installed))

    marketplace_model["versions"][0].update(version="0.2.0", commit="a" * 40, image_tag="sha-aaaaaaa")
    marketplace_model["channels"]["stable"] = "0.2.0"
    app(["update", model], result_action="return_value")
    updated = yaml.safe_load(model_deployment.overlay.read_text())
    assert updated["services"][service_name]["image"].endswith(":sha-aaaaaaa")
    assert updated["services"][service_name]["cpus"] == 2
    assert updated["services"][service_name]["volumes"] == service["volumes"]
    assert updated["volumes"] == installed["volumes"]
    assert model_deployment.runner.call_count == 4
    assert "--no-deps" in model_deployment.runner.call_args.args[0]


def test_install_local_and_update_print_url(marketplace_model, marketplace_http, model_deployment, caplog):
    model = marketplace_model["id"]
    with caplog.at_level(logging.INFO):
        app(["install", model, "--local"], result_action="return_value")
        app(["update", model, "--local"], result_action="return_value")
    config = yaml.safe_load(model_deployment.local_overlay.read_text())
    service = next(iter(config["services"].values()))
    assert service["ports"] == [{"target": 8000, "host_ip": "127.0.0.1"}]
    assert service["environment"] == {"DATABASE_URL": "sqlite+aiosqlite:////app/data/chapkit.db"}
    assert "depends_on" not in service
    assert "http://127.0.0.1:54321" in caplog.text
    assert not model_deployment.overlay.exists()


def test_custom_requires_risk_acceptance_each_time(model_deployment, caplog):
    with pytest.raises(SystemExit):
        install("custom", image="example/model:v1")
    model_deployment.runner.assert_not_called()
    assert "accept responsibility" in caplog.text
    install("custom", image="example/model:v1", accept_risk=True)
    model_deployment.runner.reset_mock()
    with pytest.raises(SystemExit):
        update("custom")
    model_deployment.runner.assert_not_called()
    update("custom", image="example/model:v2", accept_risk=True)
    update("custom", accept_risk=True)
    config = yaml.safe_load(model_deployment.overlay.read_text())
    assert config["services"]["marketplace-custom"]["image"] == "example/model:v2"
    assert config["services"]["marketplace-custom"]["x-chap-custom"] is True


def test_install_existing_and_update_missing_fail(model_deployment):
    with pytest.raises(SystemExit):
        update("custom", image="example/model:v1", accept_risk=True)
    model_deployment.runner.assert_not_called()
    install("custom", image="example/model:v1", accept_risk=True)
    model_deployment.runner.reset_mock()
    with pytest.raises(SystemExit):
        install("custom", image="example/model:v2", accept_risk=True)
    model_deployment.runner.assert_not_called()


@pytest.mark.parametrize("failure", [FileNotFoundError("docker"), subprocess.CalledProcessError(1, "docker")])
def test_failed_pull_keeps_previous_install(model_deployment, failure):
    install("custom", image="example/model:v1", accept_risk=True)
    previous = model_deployment.overlay.read_text()
    model_deployment.runner.side_effect = failure
    with pytest.raises(SystemExit):
        update("custom", image="example/model:v2", accept_risk=True)
    assert model_deployment.overlay.read_text() == previous
    assert sorted(p.name for p in model_deployment.overlay.parent.glob("*.yml")) == [
        "compose.marketplace.yml",
        "compose.yml",
    ]


def test_failed_pull_suggests_the_platform_flag(model_deployment, caplog):
    runner = model_deployment.runner.side_effect

    def fail_pull(command, **kwargs):
        if "pull" in command:
            raise subprocess.CalledProcessError(1, command)
        return runner(command, **kwargs)

    model_deployment.runner.side_effect = fail_pull
    with pytest.raises(SystemExit):
        install("custom", image="example/model:v1", accept_risk=True)
    assert "--platform linux/amd64" in caplog.text
    # Docker already printed the real reason, so the hint must be the last thing the user sees.
    assert caplog.records[-1].message.endswith("--platform linux/amd64")
    caplog.clear()
    with pytest.raises(SystemExit):
        install("custom", image="example/model:v1", accept_risk=True, platform="linux/amd64")
    assert "--platform" not in caplog.text


def test_network_failure_does_not_deploy(model_deployment, mocker):
    import httpx

    mocker.patch("httpx.Client.get", side_effect=httpx.ConnectError("Marketplace unavailable"))
    with pytest.raises(SystemExit):
        install("chapkit_simple_multistep_model")
    model_deployment.runner.assert_not_called()


def test_failed_start_restores_previous_image(model_deployment):
    install("custom", image="example/model:v1", accept_risk=True)
    previous = model_deployment.overlay.read_text()
    runner = model_deployment.runner.side_effect

    def fail_start(command, **kwargs):
        if "up" in command and str(model_deployment.overlay) not in command:
            raise subprocess.CalledProcessError(1, command)
        return runner(command, **kwargs)

    model_deployment.runner.side_effect = fail_start
    with pytest.raises(SystemExit):
        update("custom", image="example/model:v2", accept_risk=True)
    assert model_deployment.overlay.read_text() == previous
    assert model_deployment.deployments[-1]["services"]["marketplace-custom"]["image"] == "example/model:v1"
    assert "up" in model_deployment.runner.call_args.args[0]


def test_multiple_compose_files_and_platform(model_deployment, tmp_path):
    extra = tmp_path / "compose.extra.yml"
    extra.write_text("services: {}\n")
    app(
        [
            "install",
            "custom",
            "--image",
            "example/model:v1",
            "--accept-risk",
            "--compose-file",
            str(tmp_path / "compose.yml"),
            "--compose-file",
            str(extra),
            "--platform",
            "linux/amd64",
        ],
        result_action="return_value",
    )
    command = model_deployment.runner.call_args.args[0]
    assert command[:6] == ["docker", "compose", "-f", str(tmp_path / "compose.yml"), "-f", str(extra)]
    assert model_deployment.deployments[-1]["services"]["marketplace-custom"]["platform"] == "linux/amd64"


def test_invalid_registry_never_deploys(marketplace_model, marketplace_http, model_deployment):
    marketplace_model["schema_version"] = 999
    with pytest.raises(SystemExit):
        install(marketplace_model["id"])
    model_deployment.runner.assert_not_called()


def test_uninstall_removes_one_model_and_keeps_the_others(marketplace_model, marketplace_http, model_deployment):
    model = marketplace_model["id"]
    install("custom", image="example/model:v1", accept_risk=True)
    app(["install", model], result_action="return_value")
    app(["uninstall", model], result_action="return_value")
    config = yaml.safe_load(model_deployment.overlay.read_text())
    assert list(config["services"]) == ["marketplace-custom"]
    assert list(config["volumes"]) == ["marketplace-custom-data"]
    commands = [call.args[0] for call in model_deployment.runner.call_args_list]
    assert commands[-1][-4:] == ["rm", "--stop", "--force", f"marketplace-{marketplace_model['service_id']}"]
    assert not any("volume" in command for command in commands)


def test_uninstall_last_model_removes_the_overlay(model_deployment):
    install("custom", image="example/model:v1", accept_risk=True)
    uninstall("custom")
    assert not model_deployment.overlay.exists()
    assert sorted(path.name for path in model_deployment.overlay.parent.glob("*.yml")) == ["compose.yml"]


def test_uninstall_local_removes_the_local_overlay(model_deployment):
    install("custom", image="example/model:v1", accept_risk=True, local=True)
    uninstall("custom", local=True)
    assert not model_deployment.local_overlay.exists()
    assert "--project-name" in model_deployment.runner.call_args.args[0]


def test_uninstall_deletes_the_data_volume_when_asked(model_deployment):
    install("custom", image="example/model:v1", accept_risk=True)
    uninstall("custom", delete_data=True)
    assert model_deployment.runner.call_args.args[0] == [
        "docker",
        "volume",
        "rm",
        "chap-test_marketplace-custom-data",
    ]


def test_uninstall_missing_model_fails(model_deployment):
    with pytest.raises(SystemExit):
        uninstall("custom")
    model_deployment.runner.assert_not_called()


def test_failed_removal_keeps_the_model_installed(model_deployment):
    install("custom", image="example/model:v1", accept_risk=True)
    previous = model_deployment.overlay.read_text()
    model_deployment.runner.side_effect = subprocess.CalledProcessError(1, "docker")
    with pytest.raises(SystemExit):
        uninstall("custom")
    assert model_deployment.overlay.read_text() == previous

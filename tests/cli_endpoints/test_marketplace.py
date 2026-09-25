import logging
import subprocess
from types import SimpleNamespace

import pytest
import yaml
from sqlmodel import Session, SQLModel, create_engine, select

from chap_core.admin_cli import app as admin_app
from chap_core.cli import app
from chap_core.cli_endpoints import marketplace
from chap_core.cli_endpoints.marketplace import install, uninstall, update
from chap_core.database.database import SessionWrapper
from chap_core.database.model_template_seed import add_marketplace_model
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
from chap_core.services.model_marketplace import configured_model_requests, model_template_request, resolve_model


def test_resolves_stable_not_latest(marketplace_model, marketplace_http):
    marketplace_model["channels"]["latest"] = "99.0.0"
    marketplace_model["versions"].insert(
        0, {**marketplace_model["versions"][0], "version": "99.0.0", "status": "unstable"}
    )
    pin = resolve_model(marketplace_model["id"])
    assert pin.version == "0.1.0"
    assert pin.image == f"{marketplace_model['source']['image']}:sha-57eeb78"
    assert pin.commit == marketplace_model["versions"][1]["commit"]
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


@pytest.mark.parametrize("tag", ["latest", "main", "v1", "sha-0000000", "sha-", "sha-57eeb"])
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


def test_model_template_request_describes_the_pin(marketplace_model, marketplace_http):
    pin = resolve_model(marketplace_model["id"])
    request = model_template_request(pin)
    assert request["name"] == "chapkit-simple-multistep-model"
    assert request["version"] == "0.1.0"
    assert request["source_digest"] == marketplace_model["versions"][0]["commit"]
    assert request["source_url"] == "http://marketplace-chapkit-simple-multistep-model:8000"
    assert request["uses_chapkit"] is True
    assert request["display_name"] == "Simple Multistep"
    assert request["author_assessed_status"] == "orange"
    assert request["organization"] == "HISP Centre, University of Oslo"
    assert request["supported_period_type"] == "month"
    assert request["required_covariates"] == []
    assert request["allow_free_additional_continuous_covariates"] is True
    assert (request["min_prediction_periods"], request["max_prediction_periods"]) == (1, 100)


@pytest.mark.parametrize("period_types, expected", [(["weekly"], "week"), (["weekly", "monthly"], "any"), ([], "any")])
def test_model_template_request_maps_period_types(marketplace_model, marketplace_http, period_types, expected):
    marketplace_model["compatibility"]["period_types"] = period_types
    assert model_template_request(resolve_model(marketplace_model["id"]))["supported_period_type"] == expected


def test_configured_model_requests_split_out_the_reserved_config_keys(marketplace_model, marketplace_http):
    pin = resolve_model(marketplace_model["id"])
    requests = configured_model_requests(pin, 7)
    assert [r["name"] for r in requests] == ["monthly_climate", "monthly_selfhistory"]
    assert all(r["model_template_id"] == 7 for r in requests)
    climate, selfhistory = requests
    assert climate["additional_continuous_covariates"] == ["rainfall", "mean_temperature", "mean_relative_humidity"]
    assert selfhistory["additional_continuous_covariates"] == []
    assert climate["user_option_values"] == {
        "n_target_lags": 6,
        "n_samples": 100,
        "rf_max_depth": 10,
        "rf_min_samples_leaf": 5,
    }


def test_configured_model_without_covariates_gets_the_registry_defaults(marketplace_model, marketplace_http):
    del marketplace_model["configurations"]["monthly_climate"]["config"]["additional_continuous_covariates"]
    request = configured_model_requests(resolve_model(marketplace_model["id"]), 1)[0]
    assert request["additional_continuous_covariates"] == marketplace_model["covariates"]["defaults"]


def test_install_and_update_register_the_model_and_preserve_settings_and_data(
    marketplace_model, marketplace_http, model_deployment
):
    model = marketplace_model["id"]
    chap = model_deployment.chap
    admin_app(["install", model], result_action="return_value")
    installed = yaml.safe_load(model_deployment.overlay.read_text())
    service_name = f"marketplace-{marketplace_model['service_id']}"
    service = installed["services"][service_name]
    assert service["image"].endswith(":sha-57eeb78")
    assert service["environment"]["SERVICEKIT_ORCHESTRATOR_URL"].endswith("/$$register")
    assert service["environment"]["SERVICEKIT_HOST"] == service_name
    assert service["x-chap-template"] == marketplace_model["service_id"]
    # chapkit writes its database under data/ in the image's working directory.
    assert service["volumes"][0] == f"{service_name}-data:/app/data"
    assert "ports" not in service
    assert [(t["name"], t["version"], t["sourceDigest"]) for t in chap.templates] == [
        (marketplace_model["service_id"], "0.1.0", marketplace_model["versions"][0]["commit"])
    ]
    assert [(m["name"], m["modelTemplateId"]) for m in chap.configured_models] == [
        ("monthly_climate", 1),
        ("monthly_selfhistory", 1),
    ]
    service["cpus"] = 2
    model_deployment.overlay.write_text(yaml.safe_dump(installed))

    marketplace_model["versions"][0].update(version="0.2.0", commit="a" * 40, image_tag="sha-aaaaaaa")
    marketplace_model["channels"]["stable"] = "0.2.0"
    admin_app(["update", model], result_action="return_value")
    updated = yaml.safe_load(model_deployment.overlay.read_text())
    assert updated["services"][service_name]["image"].endswith(":sha-aaaaaaa")
    assert updated["services"][service_name]["cpus"] == 2
    assert updated["services"][service_name]["volumes"] == service["volumes"]
    assert updated["volumes"] == installed["volumes"]
    assert model_deployment.runner.call_count == 6
    assert "--no-deps" in model_deployment.runner.call_args.args[0]
    # The new version is a new template row with its own configurations; the old row stays.
    assert [(t["version"], t["isLive"]) for t in chap.templates] == [("0.1.0", False), ("0.2.0", True)]
    assert [m["modelTemplateId"] for m in chap.configured_models] == [1, 1, 2, 2]


def test_install_is_registered_before_docker_runs_and_can_be_repeated(marketplace_model, model_deployment):
    model = marketplace_model["id"]
    chap = model_deployment.chap
    model_deployment.runner.side_effect = subprocess.CalledProcessError(1, "docker")
    with pytest.raises(SystemExit):
        install(model)
    assert len(chap.templates) == 1 and len(chap.configured_models) == 2
    assert not model_deployment.overlay.exists()

    model_deployment.runner.side_effect = None
    install(model)
    assert len(chap.templates) == 1 and len(chap.configured_models) == 2
    assert model_deployment.overlay.exists()


def test_install_no_start_registers_and_writes_the_overlay_without_starting(marketplace_model, model_deployment):
    install(marketplace_model["id"], no_start=True)
    commands = [call.args[0] for call in model_deployment.runner.call_args_list]
    assert not any("up" in command for command in commands)
    assert len(model_deployment.chap.templates) == 1
    service = yaml.safe_load(model_deployment.overlay.read_text())["services"]
    assert list(service) == [f"marketplace-{marketplace_model['service_id']}"]


def test_install_uses_the_configured_chap_url_and_token(marketplace_model, model_deployment, monkeypatch):
    monkeypatch.setenv("CHAP_URL", "http://chap.example.org")
    monkeypatch.setenv("CHAP_API_TOKEN", "secret")
    install(marketplace_model["id"], no_start=True)
    assert len(model_deployment.chap.templates) == 1


def test_install_fails_before_docker_when_chap_is_unreachable(marketplace_model, model_deployment, caplog):
    with pytest.raises(SystemExit):
        install(marketplace_model["id"], url="http://nowhere.example.org")
    model_deployment.runner.assert_not_called()
    assert "Could not reach CHAP at http://nowhere.example.org" in caplog.text


def test_install_reports_a_refused_revision(marketplace_model, model_deployment, caplog):
    model_deployment.chap.templates.append(
        {
            "id": 1,
            "name": marketplace_model["service_id"],
            "version": "0.1.0",
            "sourceDigest": "b" * 40,
            "isLive": True,
            "archived": False,
        }
    )
    with pytest.raises(SystemExit):
        install(marketplace_model["id"])
    model_deployment.runner.assert_not_called()
    assert "409" in caplog.text and "write-once" in caplog.text


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


def test_custom_image_is_registered_from_the_running_service(model_deployment):
    chap = model_deployment.chap
    install("custom", image="example/model:v1", accept_risk=True)
    commands = [call.args[0] for call in model_deployment.runner.call_args_list]
    # The service must run before CHAP can read its info, so registration follows the start.
    assert "up" in commands[-1]
    assert [(t["name"], t["version"], t["sourceDigest"]) for t in chap.templates] == [
        ("custom-model", "1.0.0", "c" * 40)
    ]
    assert [(m["name"], m["user_option_values"]) for m in chap.configured_models] == [("default", {})]
    service = yaml.safe_load(model_deployment.overlay.read_text())["services"]["marketplace-custom"]
    assert service["x-chap-template"] == "custom-model"


def test_custom_image_that_never_registers_fails(model_deployment, monkeypatch, caplog):
    model_deployment.chap.services.clear()
    monkeypatch.setattr(marketplace, "REGISTRATION_TIMEOUT", 0)
    with pytest.raises(SystemExit):
        install("custom", image="example/model:v1", accept_risk=True)
    assert "did not register" in caplog.text
    assert not model_deployment.overlay.exists()


def test_custom_image_cannot_skip_the_start(model_deployment):
    with pytest.raises(SystemExit):
        install("custom", image="example/model:v1", accept_risk=True, no_start=True)
    model_deployment.runner.assert_not_called()


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
    assert "chap-admin install custom --platform linux/amd64" in caplog.text
    # Docker already printed the real reason, so the hint must be the last thing the user sees.
    assert caplog.records[-1].message.endswith("--platform linux/amd64")
    caplog.clear()
    with pytest.raises(SystemExit):
        install("custom", image="example/model:v1", accept_risk=True, platform="linux/amd64")
    assert "--platform" not in caplog.text


def test_network_failure_does_not_deploy(model_deployment, monkeypatch, caplog):
    monkeypatch.setenv("CHAP_MARKETPLACE_URL", "https://unreachable.example.org/registry")
    with pytest.raises(SystemExit):
        install("chapkit_simple_multistep_model", accept_risk=True)
    model_deployment.runner.assert_not_called()
    assert model_deployment.chap.requests == []
    assert "unreachable.example.org is unreachable" in caplog.text


def test_failed_start_restores_previous_image(model_deployment):
    install("custom", image="example/model:v1", accept_risk=True)
    previous = model_deployment.overlay.read_text()
    runner = model_deployment.runner.side_effect

    attempts = []

    def fail_start(command, **kwargs):
        if "up" in command:
            attempts.append(command)
            if len(attempts) == 1:
                raise subprocess.CalledProcessError(1, command)
        return runner(command, **kwargs)

    model_deployment.runner.side_effect = fail_start
    with pytest.raises(SystemExit):
        update("custom", image="example/model:v2", accept_risk=True)
    assert model_deployment.overlay.read_text() == previous
    commands = [call.args[0] for call in model_deployment.runner.call_args_list]
    # The pull has moved the tag, so the rollback must move it back before starting the overlay again.
    assert commands[-2] == ["docker", "tag", "sha256:previous", "example/model:v1"]
    assert "up" in commands[-1] and str(model_deployment.overlay) in commands[-1]


def test_multiple_compose_files_and_platform(model_deployment, tmp_path):
    extra = tmp_path / "compose.extra.yml"
    extra.write_text("services: {}\n")
    admin_app(
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
    commands = [call.args[0] for call in model_deployment.runner.call_args_list]
    assert commands[0] == ["docker", "pull", "--platform", "linux/amd64", "example/model:v1"]
    assert commands[-1][:6] == ["docker", "compose", "-f", str(tmp_path / "compose.yml"), "-f", str(extra)]
    assert model_deployment.deployments[-1]["services"]["marketplace-custom"]["platform"] == "linux/amd64"


def test_install_is_not_a_chap_command():
    with pytest.raises(SystemExit):
        app(["install", "custom"], result_action="return_value")


def test_invalid_registry_never_deploys(marketplace_model, marketplace_http, model_deployment):
    marketplace_model["schema_version"] = 999
    with pytest.raises(SystemExit):
        install(marketplace_model["id"])
    model_deployment.runner.assert_not_called()


def test_uninstall_removes_one_model_and_retires_it_in_chap(marketplace_model, marketplace_http, model_deployment):
    model = marketplace_model["id"]
    chap = model_deployment.chap
    install("custom", image="example/model:v1", accept_risk=True)
    admin_app(["install", model], result_action="return_value")
    admin_app(["uninstall", model], result_action="return_value")
    config = yaml.safe_load(model_deployment.overlay.read_text())
    assert list(config["services"]) == ["marketplace-custom"]
    assert list(config["volumes"]) == ["marketplace-custom-data"]
    commands = [call.args[0] for call in model_deployment.runner.call_args_list]
    assert commands[-1][-4:] == ["rm", "--stop", "--force", f"marketplace-{marketplace_model['service_id']}"]
    assert not any("volume" in command for command in commands)
    assert [(t["name"], t["archived"]) for t in chap.templates] == [
        ("custom-model", False),
        (marketplace_model["service_id"], True),
    ]
    assert [m["archived"] for m in chap.configured_models] == [False, True, True]
    assert ("DELETE", "/v1/crud/model-templates/2", None) in chap.requests


def test_uninstall_of_a_model_chap_does_not_list_still_removes_the_service(model_deployment, caplog):
    install("custom", image="example/model:v1", accept_risk=True)
    model_deployment.chap.templates.clear()
    with caplog.at_level(logging.INFO):
        uninstall("custom")
    assert yaml.safe_load(model_deployment.overlay.read_text())["services"] == {}
    assert "not listed by CHAP" in caplog.text


def test_uninstall_last_model_keeps_an_empty_overlay(model_deployment):
    install("custom", image="example/model:v1", accept_risk=True)
    uninstall("custom")
    # Operators were told to always pass -f compose.marketplace.yml, so it must not disappear.
    assert yaml.safe_load(model_deployment.overlay.read_text()) == {"services": {}, "volumes": {}}
    assert sorted(path.name for path in model_deployment.overlay.parent.glob("*.yml")) == [
        "compose.marketplace.yml",
        "compose.yml",
    ]


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


def test_failed_volume_removal_still_uninstalls_the_model(model_deployment, caplog):
    install("custom", image="example/model:v1", accept_risk=True)
    run = model_deployment.runner.side_effect

    def fail_volume_removal(command, **kwargs):
        if command[:3] == ["docker", "volume", "rm"]:
            return SimpleNamespace(stdout="", returncode=1)
        return run(command, **kwargs)

    model_deployment.runner.side_effect = fail_volume_removal
    # The model is already gone, so a stale volume must not leave --delete-data unretryable.
    uninstall("custom", delete_data=True)
    assert yaml.safe_load(model_deployment.overlay.read_text())["services"] == {}
    assert "Could not delete volume chap-test_marketplace-custom-data" in caplog.text


def test_update_resolves_against_the_registry_the_model_came_from(
    marketplace_model, marketplace_http, model_deployment, monkeypatch
):
    model = marketplace_model["id"]
    monkeypatch.setenv("CHAP_MARKETPLACE_URL", "https://models.example.org/registry")
    install(model, accept_risk=True)
    monkeypatch.delenv("CHAP_MARKETPLACE_URL")
    marketplace_http.clear()
    # The default marketplace must not silently take over a model installed from another registry.
    with pytest.raises(SystemExit):
        update(model)
    update(model, accept_risk=True)
    assert all(url.startswith("https://models.example.org/registry/") for url in marketplace_http)


def test_rejects_a_platform_that_breaks_compose_interpolation(model_deployment):
    with pytest.raises(SystemExit):
        install("custom", image="example/model:v1", accept_risk=True, platform="$UNSET/amd64")
    model_deployment.runner.assert_not_called()


@pytest.mark.parametrize(
    "overlay_text",
    ["services:\n  marketplace-custom:\n", "services: {}\nvolumes:\n  - bad\n"],
)
def test_rejects_a_malformed_overlay(model_deployment, overlay_text):
    model_deployment.overlay.write_text(overlay_text)
    with pytest.raises(SystemExit):
        install("custom", image="example/model:v1", accept_risk=True)
    model_deployment.runner.assert_not_called()


def test_install_mounts_the_data_volume_in_the_image_working_directory(model_deployment):
    run = model_deployment.runner.side_effect

    def scaffolded_image(command, **kwargs):
        if "{{.Config.WorkingDir}}" in command:
            return SimpleNamespace(stdout="/work\n", returncode=0)
        return run(command, **kwargs)

    model_deployment.runner.side_effect = scaffolded_image
    install("custom", image="example/model:v1", accept_risk=True)
    service = yaml.safe_load(model_deployment.overlay.read_text())["services"]["marketplace-custom"]
    assert service["volumes"][0] == "marketplace-custom-data:/work/data"


def test_overlay_stays_readable_to_other_operators(model_deployment):
    install("custom", image="example/model:v1", accept_risk=True)
    assert model_deployment.overlay.stat().st_mode & 0o044 == 0o044


def test_seeding_a_marketplace_model_stores_its_template_and_configurations(marketplace_model, marketplace_http):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        wrapper = SessionWrapper(session=session)
        template_id = add_marketplace_model(marketplace_model["id"], wrapper)
        assert add_marketplace_model(marketplace_model["id"], wrapper) == template_id
        template = session.get(ModelTemplateDB, template_id)
        assert template is not None
        assert (template.name, template.version, template.uses_chapkit) == (
            "chapkit-simple-multistep-model",
            "0.1.0",
            True,
        )
        assert template.source_digest == marketplace_model["versions"][0]["commit"]
        assert template.author_assessed_status.value == "orange"
        assert template.supported_period_type.value == "month"
        configured = session.exec(
            select(ConfiguredModelDB).where(ConfiguredModelDB.model_template_id == template_id)
        ).all()
        assert sorted(model.name for model in configured) == [
            "chapkit-simple-multistep-model:monthly_climate",
            "chapkit-simple-multistep-model:monthly_selfhistory",
        ]
        assert all(model.uses_chapkit for model in configured)
        assert configured[0].user_option_values == {
            "n_target_lags": 6,
            "n_samples": 100,
            "rf_max_depth": 10,
            "rf_min_samples_leaf": 5,
        }

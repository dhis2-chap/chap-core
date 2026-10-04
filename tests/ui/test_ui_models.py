import json
import os
from pathlib import Path

import pytest

from chap_core.services.model_marketplace import MarketplaceModel
from chap_core.ui.catalog import command_title
from chap_core.ui.models import (
    ASSESSMENT,
    ChapsError,
    ChapsModel,
    SavedModel,
    catalog_entries,
    chaps_added_models,
    chaps_models,
    chaps_project,
    chaps_registry_args,
    chaps_start,
    chaps_stop,
    forget_model,
    github_models,
    marketplace_image,
    model_label,
    newest_first,
    plain_text,
    remember_model,
    saved_models,
    start_service,
)


def test_marketplace_image_uses_the_stable_version(marketplace_model):
    entry = MarketplaceModel.model_validate(marketplace_model)
    stable = next(v for v in entry.versions if v.version == entry.channels["stable"])
    assert marketplace_image(entry) == f"{entry.source.image}:{stable.image_tag}"


def test_github_models_are_pinned_to_their_stable_commit():
    models = github_models()
    assert models
    assert all(m.url.startswith("https://github.com/") for m in models)
    pinned = [m for m in models if m.commit]
    assert all(m.model_name == f"{m.url}@{m.commit}" for m in pinned)


def test_model_label_is_the_repository_or_directory_name():
    assert model_label("https://github.com/dhis2-chap/chtorch@88f59a2") == "chtorch"
    assert model_label("external_models/naive_python_model_uv/") == "naive_python_model_uv"


def test_catalog_lists_marketplace_github_and_local_models(marketplace_model):
    entries = catalog_entries([MarketplaceModel.model_validate(marketplace_model)])
    chapkit = entries[0]
    assert chapkit.kind == "chapkit"
    assert chapkit.image == marketplace_image(MarketplaceModel.model_validate(marketplace_model))
    assert chapkit.status == ASSESSMENT[marketplace_model.get("assessed_status") or "gray"][0]
    assert {e.kind for e in entries} >= {"chapkit", "github"}
    assert all(e.model_name for e in entries if e.kind != "chapkit")


def test_command_titles_follow_page_titles():
    assert command_title("eval") == "Evaluate"
    assert command_title("generate-modelcard") == "Model card"


def test_model_label_falls_back_to_the_address_when_no_service_answers():
    assert model_label("http://localhost:1") == "localhost:1"


def test_models_you_run_are_remembered_once_and_can_be_forgotten(tmp_path):
    path = tmp_path / "models.yaml"
    remember_model(path, "https://github.com/dhis2-chap/chtorch@88f59a2")
    remember_model(path, "https://github.com/dhis2-chap/chtorch@88f59a2")
    remember_model(path, "/models/mine", "Mine")
    assert saved_models(path) == [
        SavedModel("chtorch", "https://github.com/dhis2-chap/chtorch@88f59a2"),
        SavedModel("Mine", "/models/mine"),
    ]
    forget_model(path, "/models/mine")
    assert [m.model for m in saved_models(path)] == ["https://github.com/dhis2-chap/chtorch@88f59a2"]


def test_addresses_of_chapkit_services_are_not_remembered(tmp_path):
    path = tmp_path / "models.yaml"
    remember_model(path, "http://localhost:5001")
    assert saved_models(path) == []
    assert not path.exists()


def test_saved_models_appear_in_the_catalog(tmp_path):
    path = tmp_path / "models.yaml"
    remember_model(path, "/models/mine", "Mine")
    saved = [e for e in catalog_entries([], saved_models(path)) if e.kind == "saved"]
    assert [(e.name, e.model_name) for e in saved] == [("Mine", "/models/mine")]


def test_chaps_project_is_only_a_deployment_chap_ui_starts_in(monkeypatch, tmp_path):
    monkeypatch.delenv("CHAPS_PROJECT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert chaps_project() is None  # models then start in a `chaps run` group
    (tmp_path / ".chaps").mkdir()
    (tmp_path / ".chaps" / "project.yaml").touch()
    assert chaps_project() == tmp_path.resolve()
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "sub")
    assert chaps_project() == tmp_path.resolve()  # as chaps finds it from a folder inside one


def fake_chaps(tmp_path, monkeypatch, script: str) -> Path:
    """A chaps on PATH that runs `script` and records its arguments in the returned file."""
    calls = tmp_path / "calls"
    fake = tmp_path / "bin" / "chaps"
    fake.parent.mkdir(exist_ok=True)
    fake.write_text(f'#!/bin/sh\necho "$@" >> {calls}\n{script}\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    return calls


def ps_row(service_id: str, state: str, url: str | None, project_dir) -> dict:
    port = int(url.rsplit(":", 1)[-1]) if url else None
    return {"id": service_id.replace("-", "_"), "service_id": service_id, "state": state, "url": url,
            "port": port, "project_dir": str(project_dir)}  # fmt: skip


def test_chaps_models_reads_chaps_ps(monkeypatch, tmp_path):
    listed = {
        "models": [
            ps_row("chapkit-ewars-model", "up", "http://localhost:5001", tmp_path),
            ps_row("auto-arima-chapkit", "not running", "http://localhost:5002", tmp_path),
        ]
    }
    calls = fake_chaps(tmp_path, monkeypatch, f"echo '{json.dumps(listed)}'")
    ewars, arima = chaps_models(None)
    assert ewars == ChapsModel("chapkit_ewars_model", "chapkit-ewars-model", "up", "http://localhost:5001", tmp_path)
    assert ewars.answering
    assert not arima.answering
    assert calls.read_text().split() == ["--json", "ps"]


def test_chaps_gets_the_registry_index_of_the_marketplace_in_use(monkeypatch):
    monkeypatch.delenv("CHAP_MARKETPLACE_URL", raising=False)
    assert chaps_registry_args() == []
    monkeypatch.setenv("CHAP_MARKETPLACE_URL", "https://example.org/registry")
    assert chaps_registry_args() == ["--registry-url", "https://example.org/registry/registry.yaml"]


def test_models_behind_chap_core_are_internal_until_exposed(monkeypatch, tmp_path):
    # chaps reports chap-core's read-only proxy as the address of a model without a host port.
    proxy = "http://localhost:8700/v2/services/auto-arima-chapkit/run/"
    listed = {"models": [{**ps_row("auto-arima-chapkit", "registered", None, tmp_path), "url": proxy}]}
    fake_chaps(tmp_path, monkeypatch, f"echo '{json.dumps(listed)}'")
    (model,) = chaps_models(tmp_path)
    assert model.url is None
    assert model.internal
    assert not model.answering


def test_chaps_start_runs_the_model_in_a_deployment_or_a_group(monkeypatch, tmp_path):
    calls = fake_chaps(tmp_path, monkeypatch, """echo '{"ok": true, "url": "http://localhost:5001"}'""")
    assert chaps_start("https://github.com/me/my_model", None, "my_model")["url"] == "http://localhost:5001"
    chaps_start("chapkit_ewars_model", tmp_path)
    in_group, in_deployment = calls.read_text().splitlines()
    assert in_group == "--json run --no-wait --allow-template --id my_model -- https://github.com/me/my_model"
    assert in_deployment == f"--json -C {tmp_path} run --no-wait --allow-template -- chapkit_ewars_model"


def test_chaps_stop_acts_where_the_model_runs(monkeypatch, tmp_path):
    calls = fake_chaps(tmp_path, monkeypatch, """echo '{"ok": true}'""")
    model = ChapsModel("my_model", "my-model", "up", "http://localhost:5001", tmp_path / "group")
    chaps_stop(model)
    chaps_stop(model, delete_data=True)
    keep, delete = calls.read_text().splitlines()
    assert keep.split() == ["--json", "-C", str(tmp_path / "group"), "stop", "my_model"]
    assert delete.split()[-1] == "--purge"


def test_chaps_failures_report_chaps_error(monkeypatch, tmp_path):
    failure = {
        "ok": False,
        "error": "nothing did not start (denied)",
        "hint": "fix that, then `chaps run x` tries again",
    }
    fake_chaps(tmp_path, monkeypatch, f"echo 'Pulling' >&2; echo '{json.dumps(failure)}'; exit 1")
    with pytest.raises(ChapsError) as raised:
        chaps_start("ghcr.io/nobody/nothing:1", None)
    assert str(raised.value) == "nothing did not start (denied); fix that, then `chaps run x` tries again"
    assert "Pulling" in raised.value.output


def test_running_models_started_from_a_url_are_catalog_entries(monkeypatch, tmp_path):
    listed = [
        {"id": "chapkit_ewars_model", "service_id": "chapkit-ewars-model", "image": "ghcr.io/a/b:1", "manual": False,
         "enabled": True},
        {"id": "my_model", "service_id": "my-model", "display_name": "My model", "image": "ghcr.io/me/my_model:sha-1",
         "manual": True, "enabled": True},
        {"id": "stopped", "service_id": "stopped", "image": "ghcr.io/me/stopped:1", "manual": True, "enabled": False},
    ]  # fmt: skip
    fake_chaps(tmp_path, monkeypatch, f"echo '{json.dumps(listed)}'")
    running = [ChapsModel("my_model", "my-model", "up", "http://localhost:5001", tmp_path)]
    (entry,) = chaps_added_models(running)
    assert (entry.id, entry.name, entry.kind, entry.service_id) == ("my_model", "My model", "chapkit", "my-model")


def test_the_same_model_in_two_groups_is_two_instances(monkeypatch, tmp_path):
    listed = {
        "models": [
            {**ps_row("auto-arima-chapkit", "up", "http://localhost:5001", tmp_path / "default"), "group": "default"},
            {**ps_row("auto-arima-chapkit", "up", "http://localhost:5002", tmp_path / "other"), "group": "other"},
        ]
    }
    fake_chaps(tmp_path, monkeypatch, f"echo '{json.dumps(listed)}'")
    assert [(m.group, m.url) for m in chaps_models(None)] == [
        ("default", "http://localhost:5001"),
        ("other", "http://localhost:5002"),
    ]


def test_chaps_that_does_not_answer_lists_nothing(monkeypatch, tmp_path):
    import subprocess

    from chap_core.ui import services

    fake_chaps(tmp_path, monkeypatch, "true")

    def hang(argv, timeout=1800):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(services, "run_external", hang)
    assert chaps_models(None) == []


def test_starting_again_replaces_a_container_that_exited(monkeypatch):
    import docker

    from chap_core.ui.models import SERVICE_LABEL, start_service

    class Container:
        def __init__(self, status):
            self.status, self.removed = status, False
            self.id, self.name = "abc", "chap-auto-arima-chapkit"
            self.image = type("Image", (), {"tags": ["img"], "short_id": "img"})()
            self.labels = {SERVICE_LABEL: "auto_arima_chapkit"}
            self.ports = {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5005"}]}

        def remove(self):
            self.removed = True

        def reload(self):
            pass

    exited = Container("exited")

    class Containers:
        def list(self, all, filters):
            return [exited]

        def run(self, image, **kwargs):
            assert exited.removed, "the exited container still holds the name"
            return Container("running")

    monkeypatch.setattr(docker, "from_env", lambda: type("Client", (), {"containers": Containers()})())
    assert start_service("img", "auto_arima_chapkit").status == "running"


def test_metrics_are_read_from_chapkits_prometheus_output():
    from chap_core.ui.models import parse_metrics

    text = """# HELP ml_train_jobs_total Total number of ML training jobs submitted
# TYPE ml_train_jobs_total counter
ml_train_jobs_total 2.0
ml_predict_jobs_total 3.0
http_server_duration_milliseconds_count{http_method="GET",http_target="/health"} 4.0
http_server_duration_milliseconds_count{http_method="POST",http_target="/api/v1/ml/$train"} 2.0
process_resident_memory_bytes 1.42508032e+08
"""
    metrics = parse_metrics(text)
    assert (metrics.trainings, metrics.predictions, metrics.requests) == (2.0, 3.0, 6.0)
    assert metrics.memory_bytes == 142508032.0
    assert metrics.cpu_seconds is None  # not every service reports it


def test_a_service_without_metrics_reports_none():
    from chap_core.ui.models import service_metrics

    assert service_metrics("http://127.0.0.1:9") is None


def test_request_totals_are_read_under_servicekit_3s_metric_names():
    from chap_core.ui.models import parse_metrics

    text = 'http_server_request_duration_seconds_count{http_request_method="GET",http_route="/health"} 7.0\n'
    assert parse_metrics(text).requests == 7.0


def test_log_text_loses_terminal_colour_codes():
    assert plain_text("\x1b[2m2026-10-04\x1b[0m [\x1b[32m\x1b[1minfo\x1b[0m] ready") == "2026-10-04 [info] ready"


def test_an_image_without_a_native_build_starts_under_amd64_emulation(monkeypatch):
    import docker

    class Container:
        id, name, status, labels, ports = "c1", "chap-m", "running", {}, {"8000/tcp": [{"HostPort": "5009"}]}
        image = type("Image", (), {"tags": ["img:1"], "short_id": "i"})()

        def reload(self):
            pass

    calls = []

    class Client:
        class containers:
            @staticmethod
            def list(**kwargs):
                return []

            @staticmethod
            def run(image, **kwargs):
                calls.append(("run", kwargs.get("platform")))
                if "platform" not in kwargs:
                    raise docker.errors.ImageNotFound("no matching manifest for linux/arm64/v8 in the manifest list")
                return Container()

        class images:
            @staticmethod
            def pull(image, platform=None):
                calls.append(("pull", platform))

    monkeypatch.setattr(docker, "from_env", lambda: Client())
    service = start_service("img:1", "m")
    assert calls == [("run", None), ("pull", "linux/amd64"), ("run", "linux/amd64")]
    assert service.url == "http://localhost:5009"


def test_logs_show_the_newest_line_first_without_routine_requests():
    logs = 'starting\n"GET /health HTTP/1.1" 200\ntrained\npath=/api/v1/info status_code=200\npredicted'
    assert newest_first(logs) == "predicted\ntrained\nstarting"


def test_a_github_repository_that_does_not_exist_is_said_in_words():
    error = ChapsError("HTTP 404 from https://api.github.com/repos/nope-org/no-model", "the full output")
    assert str(error) == "GitHub has no repository nope-org/no-model, or it is private: check the URL"
    assert error.output == "the full output"

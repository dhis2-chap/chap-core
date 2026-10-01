import json
import os

from chap_core.services.model_marketplace import MarketplaceModel
from chap_core.ui.catalog import command_title
from chap_core.ui.models import (
    ASSESSMENT,
    ChapsModel,
    SavedModel,
    catalog_entries,
    chaps_models,
    chaps_project,
    forget_model,
    github_models,
    marketplace_image,
    model_label,
    remember_model,
    saved_models,
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


def test_chaps_project_is_found_from_the_folder_chap_ui_starts_in(monkeypatch, tmp_path):
    monkeypatch.delenv("CHAPS_PROJECT_DIR", raising=False)
    monkeypatch.setenv("CHAP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.chdir(tmp_path)
    assert chaps_project() is None
    (tmp_path / ".chaps").mkdir()
    assert chaps_project() == tmp_path


def test_chaps_models_reads_the_deployment_status(monkeypatch, tmp_path):
    status = {
        "models": [
            {"id": "chapkit-ewars-model", "state": "up", "reach": "http://localhost:5001"},
            {"id": "auto-arima-chapkit", "state": "not-running", "reach": "http://localhost:5004"},
        ]
    }
    fake = tmp_path / "bin" / "chaps"
    fake.parent.mkdir()
    fake.write_text(f"#!/bin/sh\necho '{json.dumps(status)}'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    models = chaps_models(tmp_path)
    assert models["chapkit-ewars-model"] == ChapsModel("chapkit-ewars-model", "up", "http://localhost:5001")
    assert models["chapkit-ewars-model"].answering
    assert not models["auto-arima-chapkit"].answering

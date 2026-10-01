from chap_core.services.model_marketplace import MarketplaceModel
from chap_core.ui.catalog import command_title
from chap_core.ui.models import ASSESSMENT, catalog_entries, github_models, marketplace_image, model_label


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

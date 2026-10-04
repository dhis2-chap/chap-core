import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import chap_core.ui
from chap_core.ui.catalog import COMMAND_PAGES

UI = Path(chap_core.ui.__file__).parent
VIEWS = UI / "views"


@pytest.fixture
def workdir(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAP_RUNS_DIR", str(tmp_path))
    return tmp_path


def render_command_page(command):
    from chap_core.ui.widgets import command_form, run_panel

    fields, values = command_form(command)
    run_panel(command, fields, values, label=command)


@pytest.mark.parametrize("command", [page.command for pages in COMMAND_PAGES.values() for page in pages])
def test_command_page_renders_form_and_cli_command(workdir, command):
    at = AppTest.from_function(render_command_page, args=(command,), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.code[0].value.startswith(f"chap {command}")


def test_app_starts_on_the_use_cases(workdir):
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.title[0].value == "What do you want to do?"
    assert any(button.label == "Start" for button in at.button)


def test_the_guide_says_when_no_model_fits(workdir, monkeypatch, data_path):
    import streamlit as st

    monkeypatch.setenv("CHAP_MARKETPLACE_URL", "http://127.0.0.1:9")  # unreachable: no network in the test
    st.cache_data.clear()
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    at.session_state["guide"] = {"step": 2, "dataset_csv": str(data_path / "laos_subset.csv"), "dataset_name": "Laos"}
    at.switch_page("views/guide.py")
    at.run()
    assert not at.exception
    assert any("None of the marketplace models can use this data" in w.value for w in at.warning)


def test_the_guide_drops_saved_answers_that_are_no_longer_offered(workdir, monkeypatch, data_path):
    import streamlit as st

    monkeypatch.setenv("CHAP_MARKETPLACE_URL", "http://127.0.0.1:9")  # unreachable: no network in the test
    st.cache_data.clear()
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    at.session_state["guide"] = {
        "step": 3,
        "dataset_csv": str(data_path / "laos_subset.csv"),
        "dataset_name": "Laos",
        "models": [],
        "horizon_n": 12,
        "thoroughness": "Exhaustive",
    }
    at.switch_page("views/guide.py")
    at.run()
    assert not at.exception
    assert at.session_state["guide-horizon"] == 3
    assert at.session_state["guide-thoroughness"] == "Normal"


def test_results_view_shows_example_evaluation(workdir, data_path):
    at = AppTest.from_file(str(VIEWS / "results.py"), default_timeout=120)
    at.session_state["selected_evals"] = [str(data_path / "example_evaluation.nc")]
    at.run()
    assert not at.exception
    assert at.multiselect[0].value == [str(data_path / "example_evaluation.nc")]
    headline = at.dataframe[0].value
    assert list(headline.columns) == ["Evaluation", "CRPS", "MAE", "RMSE", "Within 50% interval"]
    assert len(headline) == 1


def test_evaluate_view_uses_the_selected_dataset(workdir, data_path):
    at = AppTest.from_file(str(VIEWS / "evaluate.py"), default_timeout=60)
    at.session_state["dataset_csv"] = str(data_path / "laos_subset.csv")
    at.run()
    assert not at.exception
    assert at.code[0].value.startswith("chap eval --model-name")
    assert str(data_path / "laos_subset.csv") in at.code[0].value


def test_runs_view_without_runs(workdir):
    at = AppTest.from_file(str(VIEWS / "runs.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.info[0].value == "Nothing has been run yet."


def test_configure_view_starts_from_the_selected_model(workdir):
    at = AppTest.from_file(str(VIEWS / "configure.py"), default_timeout=60)
    at.session_state["model_name"] = "https://github.com/dhis2-chap/chtorch"
    at.run()
    assert not at.exception
    assert at.text_input[0].value == "https://github.com/dhis2-chap/chtorch"


def test_configure_view_can_stop_using_the_configuration(workdir, data_path):
    at = AppTest.from_file(str(VIEWS / "configure.py"), default_timeout=60)
    at.session_state["model_configuration_yaml"] = str(data_path / "hpo_config.yaml")
    at.run()
    next(button for button in at.button if button.label == "Stop using it").click()
    at.run()
    assert not at.exception
    assert at.session_state["model_configuration_yaml"] is None
    assert at.info[0].value.startswith("No model configuration is in use")


def test_data_view_keeps_a_dataset_chosen_on_another_page(workdir, data_path):
    nicaragua = str(data_path / "nicaragua_weekly_subset.csv")
    at = AppTest.from_file(str(VIEWS / "data.py"), default_timeout=60)
    at.session_state["dataset_csv"] = nicaragua
    at.run()
    assert not at.exception
    assert at.session_state["dataset_csv"] == nicaragua
    assert at.selectbox[0].value == "nicaragua_weekly_subset.csv (chap-core checkout)"


def test_results_view_shows_the_selection_kept_in_session_state(workdir, data_path):
    second = str(data_path / "example_evaluation_2.nc")
    at = AppTest.from_file(str(VIEWS / "results.py"), default_timeout=120)
    at.session_state["results-selected"] = [second, str(data_path / "missing.nc")]
    at.run()
    assert not at.exception
    assert at.multiselect[0].value == [second]


def test_data_view_starts_from_the_first_published_example(workdir, monkeypatch):
    monkeypatch.setattr(
        "chap_core.ui.services.fetch_published_dataset", lambda uploads_dir, url: uploads_dir / url.rsplit("/", 1)[-1]
    )
    at = AppTest.from_file(str(VIEWS / "data.py"), default_timeout=60)
    at.run()
    assert at.selectbox[0].value == "Laos, provinces, monthly"


def test_catalog_shows_how_to_install_chaps_when_it_is_missing(workdir, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "no-programs"))
    monkeypatch.setenv("CHAP_MARKETPLACE_URL", "http://127.0.0.1:9")  # unreachable: no network in the test
    at = AppTest.from_file(str(VIEWS / "models.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("winterop-com/chaps/main/install.sh" in block.value for block in at.code)


def catalog_with_chaps(monkeypatch, tmp_path, rows: list[dict]) -> AppTest:
    """The catalog page with a chaps on PATH whose `ps` lists `rows`."""
    import json

    import streamlit as st

    ps = json.dumps({"models": rows})
    fake = tmp_path / "bin" / "chaps"
    fake.parent.mkdir(exist_ok=True)
    fake.write_text(f"#!/bin/sh\ncase \"$*\" in\n  *' ps') echo '{ps}' ;;\n  *) echo '[]' ;;\nesac\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    monkeypatch.setenv("CHAP_MARKETPLACE_URL", "http://127.0.0.1:9")  # unreachable: no network in the test
    monkeypatch.chdir(tmp_path)
    st.cache_data.clear()  # what other tests asked chaps is cached across AppTests
    at = AppTest.from_file(str(VIEWS / "models.py"), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def ps_row(state: str, group: str | None, project_dir) -> dict:
    return {"id": "my_model", "service_id": "my-model", "state": state, "url": "http://127.0.0.1:9", "port": 9,
            "group": group, "project_dir": str(project_dir)}  # fmt: skip


def test_catalog_lists_instances_chaps_runs_with_a_stop_menu(workdir, monkeypatch, tmp_path):
    at = catalog_with_chaps(monkeypatch, tmp_path, [ps_row("up", "default", tmp_path)])
    assert any("**my_model**" in block.value for block in at.markdown)
    assert {"Stop, keep its data", "Stop and delete its data", "Stop all"} <= {b.label for b in at.button}


def test_a_stopped_model_can_be_started_again(workdir, monkeypatch, tmp_path):
    at = catalog_with_chaps(monkeypatch, tmp_path, [ps_row("not running", "default", tmp_path)])
    assert "Start" in {b.label for b in at.button}


def test_stop_all_leaves_a_deployments_models_alone(workdir, monkeypatch, tmp_path):
    (tmp_path / ".chaps").mkdir()
    (tmp_path / ".chaps" / "project.yaml").touch()
    at = catalog_with_chaps(monkeypatch, tmp_path, [ps_row("up", None, tmp_path)])
    assert "Stop all" not in {b.label for b in at.button}


def test_every_page_has_the_command_palette_shortcut(workdir):
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("chapCommandPalette" in str(block.proto) for block in at.get("html"))


def test_deleting_data_is_only_offered_for_chap_uis_own_group(workdir, monkeypatch, tmp_path):
    rows = [
        ps_row("up", "default", tmp_path / "default"),
        {**ps_row("up", "other", tmp_path / "other"), "id": "theirs"},
    ]
    at = catalog_with_chaps(monkeypatch, tmp_path, rows)
    purge_buttons = [b for b in at.button if b.label == "Stop and delete its data"]
    assert len(purge_buttons) == 1

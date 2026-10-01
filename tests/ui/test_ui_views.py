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
    monkeypatch.setenv("CHAP_UI_WORKDIR", str(tmp_path))
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


def test_app_starts_on_the_data_page(workdir):
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Dataset"


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
    assert str(at.selectbox[0].value) == nicaragua


def test_results_view_shows_the_selection_kept_in_session_state(workdir, data_path):
    second = str(data_path / "example_evaluation_2.nc")
    at = AppTest.from_file(str(VIEWS / "results.py"), default_timeout=120)
    at.session_state["results-selected"] = [second, str(data_path / "missing.nc")]
    at.run()
    assert not at.exception
    assert at.multiselect[0].value == [second]

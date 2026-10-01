from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import chap_core.ui

VIEWS = Path(chap_core.ui.__file__).parent / "views"


def test_results_view_shows_example_evaluation(monkeypatch, tmp_path, data_path):
    monkeypatch.setenv("CHAP_UI_WORKDIR", str(tmp_path))
    at = AppTest.from_file(str(VIEWS / "results.py"), default_timeout=120)
    at.session_state["selected_evals"] = [str(data_path / "example_evaluation.nc")]
    at.run()
    assert not at.exception
    assert at.multiselect[0].value == [str(data_path / "example_evaluation.nc")]
    assert len(at.dataframe) == 1


def test_evaluate_view_shows_cli_command_for_selected_dataset(monkeypatch, tmp_path, data_path):
    monkeypatch.setenv("CHAP_UI_WORKDIR", str(tmp_path))
    at = AppTest.from_file(str(VIEWS / "evaluate.py"), default_timeout=60)
    at.session_state["dataset_csv"] = str(data_path / "laos_subset.csv")
    at.run()
    assert not at.exception
    assert at.code[0].value.startswith("chap eval --model-name")
    assert str(data_path / "laos_subset.csv") in at.code[0].value

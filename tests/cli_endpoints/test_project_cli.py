import json
import logging
from pathlib import Path

import pytest
import yaml
from cyclopts import CycloptsError

from chap_core.cli import app
from chap_core.cli_endpoints.init import init_cmd
from chap_core.project import Project, model_label


def parse(*tokens: str):
    """The command and the arguments it would be called with, after project defaults are applied."""
    command, bound, _ = app.parse_args(list(tokens), exit_on_error=False, print_error=False)
    return command, bound.arguments


def edit_project_file(root: Path, **settings):
    path = root / "chap.yaml"
    path.write_text(yaml.safe_dump({**yaml.safe_load(path.read_text()), **settings}))


def test_init_creates_the_project_layout(chap_project):
    settings = yaml.safe_load((chap_project / "chap.yaml").read_text())
    assert settings["model_name"] == "builtin:seasonal_median"
    assert settings["eval"]["backtest_params"]["n_splits"] == 2
    assert (chap_project / "input" / "data.csv").is_file()
    assert (chap_project / "input" / "data.geojson").is_file()
    assert (chap_project / "output").is_dir()
    assert ".chap/" in (chap_project / ".gitignore").read_text().splitlines()


def test_init_refuses_an_existing_project_without_force(chap_project):
    with pytest.raises(ValueError, match="already exists"):
        init_cmd(chap_project)
    init_cmd(chap_project, model="builtin:persistence", force=True)
    assert Project.load(chap_project).settings["model_name"] == "builtin:persistence"


def test_init_copies_configuration_files_to_their_fixed_names(tmp_path):
    data_settings = tmp_path / "masks.json"
    data_settings.write_text(json.dumps({"maskedRanges": [{"start": "2011-06", "end": "2011-08"}]}))
    model_config = tmp_path / "options.yaml"
    model_config.write_text("user_option_values: {}\n")
    init_cmd(tmp_path / "project", data_settings=data_settings, model_config=model_config)
    input_dir = tmp_path / "project" / "input"
    assert yaml.safe_load((input_dir / "data_settings.yaml").read_text())["maskedRanges"][0]["start"] == "2011-06"
    assert (input_dir / "model_config.yaml").read_text() == "user_option_values: {}\n"


def test_eval_runs_with_no_options_inside_a_project(chap_project, monkeypatch):
    subdirectory = chap_project / "notes"
    subdirectory.mkdir()
    monkeypatch.chdir(subdirectory)
    command, arguments = parse("eval")
    command(**arguments)
    outputs = sorted(path.name for path in (chap_project / "output").iterdir())
    assert len(outputs) == 2
    assert outputs[0].startswith("seasonal_median-")
    assert outputs[0].endswith(".html")
    assert outputs[1].endswith(".nc")


def test_command_line_overrides_project_values_field_by_field(chap_project, monkeypatch):
    monkeypatch.chdir(chap_project)
    _, arguments = parse("eval", "--backtest-params.n-periods", "2")
    assert arguments["backtest_params"].n_periods == 2
    assert arguments["backtest_params"].n_splits == 2
    assert arguments["dataset_csv"] == str(chap_project / "input" / "data.csv")
    assert arguments["run_config"].runs_dir == chap_project / ".chap" / "runs"


def test_command_section_overrides_shared_keys(chap_project, monkeypatch):
    edit_project_file(chap_project, historical_context_years=2)
    monkeypatch.chdir(chap_project)
    assert parse("eval")[1]["historical_context_years"] == 2
    edit_project_file(chap_project, eval={"historical_context_years": 3})
    assert parse("eval")[1]["historical_context_years"] == 3


def test_input_files_are_skipped_when_their_owner_is_given(chap_project, monkeypatch, laos_csv):
    input_dir = chap_project / "input"
    (input_dir / "data_settings.yaml").write_text("maskedRanges: []\n")
    (input_dir / "model_config.yaml").write_text("user_option_values: {}\n")
    monkeypatch.chdir(chap_project)

    _, arguments = parse("eval")
    assert arguments["data_configuration"] == input_dir / "data_settings.yaml"
    assert arguments["model_configuration_yaml"] == input_dir / "model_config.yaml"

    _, arguments = parse("eval", "--dataset-csv", str(laos_csv), "--model-name", "builtin:persistence")
    assert "data_configuration" not in arguments
    assert "model_configuration_yaml" not in arguments
    assert arguments["output_file"].name.startswith("persistence-")


def test_relative_paths_in_the_project_file_are_relative_to_the_project(chap_project, monkeypatch):
    edit_project_file(chap_project, dataset_csv="../shared.csv")
    (chap_project / "notes").mkdir()
    monkeypatch.chdir(chap_project / "notes")
    _, arguments = parse("validate")
    assert arguments["dataset_csv"] == str(chap_project / "../shared.csv")
    assert arguments["model_name"] == "builtin:seasonal_median"


def test_misspelled_nested_key_is_an_error(chap_project, monkeypatch):
    edit_project_file(chap_project, backtest_params={"n_period": 6})
    monkeypatch.chdir(chap_project)
    with pytest.raises(ValueError, match="backtest_params.n_period"):
        parse("eval")


def test_key_no_command_takes_is_a_warning(chap_project, monkeypatch, caplog):
    edit_project_file(chap_project, backtest_param={"n_periods": 6})
    monkeypatch.chdir(chap_project)
    with caplog.at_level(logging.WARNING):
        parse("eval")
    assert "backtest_param" in caplog.text


def test_nothing_is_filled_in_outside_a_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(CycloptsError):
        parse("eval")


@pytest.mark.parametrize(
    ("source", "label"),
    [
        ("builtin:seasonal_median@1", "seasonal_median"),
        ("https://github.com/org/repo.git@abc123", "repo"),
        ("./models/my_model/", "my_model"),
        ("http://localhost:8000", "localhost-8000"),
    ],
)
def test_model_label(source, label):
    assert model_label(source) == label

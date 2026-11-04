import json
import logging
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import git
import pytest
import yaml
from cyclopts import CycloptsError

from chap_core.api_types import RunConfig
from chap_core.cli import app
from chap_core.cli_endpoints.init import init_cmd
from chap_core.project import Project, find_project, model_label


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


def test_init_writes_a_data_settings_example_that_sets_nothing(chap_project):
    from chap_core.cli_endpoints._common import load_data_configuration

    data_settings = chap_project / "input" / "data_settings.yaml"
    assert "maskedRanges" in data_settings.read_text()
    settings = load_data_configuration(data_settings)
    assert settings is not None
    assert settings.masked_ranges == []
    assert not (chap_project / "input" / "column_mapping.json").exists()


def test_init_with_force_keeps_edited_data_settings(chap_project):
    data_settings = chap_project / "input" / "data_settings.yaml"
    data_settings.write_text('maskedRanges:\n  - start: "2011-06"\n    end: "2011-08"\n')
    init_cmd(chap_project, model="builtin:persistence", force=True)
    assert "2011-06" in data_settings.read_text()


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


def test_init_stores_a_model_next_to_the_project_relative_to_it_with_its_run_options(
    tmp_path, models_path, monkeypatch
):
    shutil.copytree(models_path / "naive_python_model_with_mlproject_file", tmp_path / "models" / "naive")
    monkeypatch.chdir(tmp_path / "models")
    init_cmd(Path("../laos"), model="naive", run_config=RunConfig(ignore_environment=True))
    settings = Project.load(tmp_path / "laos").settings
    assert settings["model_name"] == "../models/naive"
    assert settings["run_config"] == {"ignore_environment": True}


def test_init_stores_a_model_further_away_as_an_absolute_path(tmp_path, models_path):
    model = models_path / "naive_python_model_with_mlproject_file"
    init_cmd(tmp_path / "studies" / "laos", model=str(model), run_config=RunConfig(ignore_environment=True))
    assert Project.load(tmp_path / "studies" / "laos").settings["model_name"] == str(model.resolve())


def test_validate_leaves_no_model_copy_behind(chap_project, models_path, monkeypatch):
    monkeypatch.chdir(chap_project)
    command, arguments = parse("validate", "--model-name", str(models_path / "naive_python_model_with_mlproject_file"))
    command(**arguments)
    assert not (chap_project / "runs").exists()
    assert list((chap_project / ".chap" / "runs").iterdir()) == []


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


def test_model_run_options_are_dropped_when_another_model_is_given(chap_project, monkeypatch):
    edit_project_file(chap_project, run_config={"is_chapkit_model": True, "as_chapkit": True})
    monkeypatch.chdir(chap_project)
    assert parse("eval")[1]["run_config"].is_chapkit_model
    run_config = parse("eval", "--model-name", "builtin:persistence")[1]["run_config"]
    assert not run_config.is_chapkit_model
    assert not run_config.as_chapkit
    assert run_config.runs_dir == chap_project / ".chap" / "runs"
    given = parse("eval", "--model-name", "../other", "--run-config.is-chapkit-model")[1]["run_config"]
    assert given.is_chapkit_model
    assert not given.as_chapkit


def test_runs_started_in_the_same_second_get_different_output_files(chap_project, monkeypatch):
    monkeypatch.chdir(chap_project)
    first, second = parse("eval")[1]["output_file"], parse("eval")[1]["output_file"]
    assert first != second
    assert first.name.startswith("seasonal_median-")


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
        ("docker://ghcr.io/chap-models/chapkit_ghr_model:sha-84fbe77", "chapkit_ghr_model"),
    ],
)
def test_model_label(source, label):
    assert model_label(source) == label


def test_init_stores_a_chap_hub_model_as_its_stable_image(marketplace_model):
    from chap_core.cli_endpoints.init import resolve_model_source
    from chap_core.services.model_marketplace import MarketplaceModel, stable_pin

    hub_pin = stable_pin(MarketplaceModel.model_validate(marketplace_model))
    with patch("chap_core.services.model_marketplace.resolve_model", return_value=hub_pin) as resolve:
        source = resolve_model_source(hub_pin.entry.id, pin=True)
    resolve.assert_called_once_with(hub_pin.entry.id)
    assert source == f"docker://{hub_pin.image}"


def test_init_refuses_a_name_that_is_neither_a_directory_nor_a_chap_hub_model():
    from chap_core.cli_endpoints.init import resolve_model_source

    unlisted = ValueError("Model 'no_such_model' is not listed in the marketplace.")
    with (
        patch("chap_core.services.model_marketplace.resolve_model", side_effect=unlisted),
        pytest.raises(ValueError, match="not a local directory, and not a Chap Hub model"),
    ):
        resolve_model_source("no_such_model", pin=True)


def test_init_pins_a_github_model_unless_told_not_to():
    from chap_core.cli_endpoints.init import resolve_model_source

    url = "https://github.com/chap-models/some_model"
    with patch("chap_core.cli_endpoints.init.remote_head_commit", return_value="a" * 40):
        assert resolve_model_source(url, pin=True) == f"{url}@{'a' * 40}"
        assert resolve_model_source(f"{url}@{'b' * 40}", pin=True) == f"{url}@{'b' * 40}"
    assert resolve_model_source(url, pin=False) == url


def test_remote_head_commit_reads_the_default_branch(tmp_path):
    from chap_core.cli_endpoints.init import remote_head_commit

    repository = git.Repo.init(tmp_path)
    (tmp_path / "MLproject").write_text("name: model\n")
    repository.index.add(["MLproject"])
    commit = repository.index.commit("Add model")
    assert remote_head_commit(str(tmp_path)) == commit.hexsha


def test_init_with_force_keeps_model_options_for_the_same_model_and_drops_them_for_another(tmp_path, models_path):
    model = str(models_path / "naive_python_model_with_mlproject_file_and_docker")
    root = tmp_path / "project"
    model_config = root / "input" / "model_config.yaml"
    init_cmd(root, model=model)
    assert yaml.safe_load(model_config.read_text())["user_option_values"] == {"some_option": "10"}

    model_config.write_text("user_option_values:\n  some_option: '99'\n")
    init_cmd(root, model=model, force=True)
    assert "'99'" in model_config.read_text()

    init_cmd(root, model="builtin:persistence", force=True)
    assert not model_config.exists()


def test_project_search_stops_at_a_git_repository_root(tmp_path):
    (tmp_path / "chap.yaml").write_text("model_name: builtin:persistence\n")
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    (tmp_path / "repo" / "src").mkdir()
    assert find_project(tmp_path / "repo" / "src") is None
    project = find_project(tmp_path)
    assert project is not None
    assert project.root == tmp_path.resolve()


def test_project_file_owned_by_another_user_is_ignored(chap_project):
    with patch("os.getuid", return_value=os.getuid() + 1):
        assert find_project(chap_project) is None


def test_dataset_files_in_input_are_skipped_when_the_project_file_names_another_dataset(
    chap_project, monkeypatch, laos_csv
):
    input_dir = chap_project / "input"
    (input_dir / "data_settings.yaml").write_text("maskedRanges: []\n")
    monkeypatch.chdir(chap_project)
    edit_project_file(chap_project, dataset_csv=str(laos_csv))
    assert "data_configuration" not in parse("validate")[1]
    edit_project_file(chap_project, dataset_csv="input/data.csv")
    assert parse("validate")[1]["data_configuration"] == input_dir / "data_settings.yaml"


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"run_config": True}, "'run_config' in chap.yaml must be a mapping"),
        ({"model": "builtin:persistence"}, "To choose the model, write model_name"),
    ],
)
def test_single_value_where_options_are_expected_is_an_error(chap_project, monkeypatch, settings, message):
    edit_project_file(chap_project, **settings)
    monkeypatch.chdir(chap_project)
    with pytest.raises(ValueError, match=message):
        parse("eval")


def test_init_refuses_to_serve_a_docker_image_with_as_chapkit(tmp_path):
    with pytest.raises(ValueError, match="already a chapkit service"):
        init_cmd(tmp_path / "project", model="docker://example/model:1", run_config=RunConfig(as_chapkit=True))
    assert not (tmp_path / "project").exists()


def test_init_writes_model_options_with_their_descriptions(tmp_path, models_path):
    root = tmp_path / "project"
    init_cmd(root, model=str(models_path / "naive_python_model_with_mlproject_file_and_docker"))
    text = (root / "input" / "model_config.yaml").read_text()
    assert yaml.safe_load(text)["user_option_values"] == {"some_option": "10"}
    assert "# Options for " in text
    assert '  # Some option for the model\n  # Type: integer. Default: "10".\n' in text


def test_init_refuses_model_options_the_model_does_not_have(tmp_path, models_path):
    model_config = tmp_path / "options.yaml"
    model_config.write_text("user_option_values:\n  some_optoin: '5'\n")
    model = str(models_path / "naive_python_model_with_mlproject_file_and_docker")
    with pytest.raises(ValueError, match="some_optoin. Its options: some_option"):
        init_cmd(tmp_path / "project", model=model, model_config=model_config)
    assert not (tmp_path / "project").exists()


def test_init_refuses_a_file_that_is_not_a_chap_dataset(tmp_path):
    not_a_dataset = tmp_path / "data.csv"
    not_a_dataset.write_text("periodname\tvalue\n2020-01\t3\n")
    with pytest.raises(ValueError, match="Could not load"):
        init_cmd(tmp_path / "project", model="builtin:persistence", dataset=str(not_a_dataset))
    assert not (tmp_path / "project").exists()


def test_init_with_a_new_dataset_checks_the_kept_data_settings(chap_project, nicaragua_path):
    (chap_project / "input" / "data_settings.yaml").write_text(
        'maskedRanges:\n  - start: "2011-06"\n    end: "2011-08"\n'
    )
    with pytest.raises(ValueError, match="data_settings.yaml"):
        init_cmd(chap_project, model="builtin:persistence", dataset=str(nicaragua_path), force=True)


def test_init_without_a_dataset_does_not_ask_for_one_the_project_has(chap_project, capsys):
    init_cmd(chap_project, model="builtin:persistence", force=True)
    assert "copy your dataset" not in capsys.readouterr().out


def test_init_refuses_to_serve_a_model_whose_environment_chapkit_cannot_set_up(tmp_path, models_path):
    from chap_core.exceptions import InvalidModelException

    model = str(models_path / "naive_python_model_with_mlproject_file")
    with pytest.raises(InvalidModelException, match="python_env"):
        init_cmd(tmp_path / "project", model=model, run_config=RunConfig(as_chapkit=True))
    assert not (tmp_path / "project").exists()


def test_commands_that_run_the_model_use_the_projects_runs_directory(chap_project, monkeypatch):
    monkeypatch.chdir(chap_project)
    for command in ("eval", "sanity-check-model"):
        assert parse(command)[1]["run_config"].runs_dir == chap_project / ".chap" / "runs"


def test_sanity_check_model_runs_with_no_options_inside_a_project(chap_project, monkeypatch, capsys):
    monkeypatch.chdir(chap_project)
    command, arguments = parse("sanity-check-model")
    assert arguments["model_name"] == "builtin:seasonal_median"
    assert arguments["dataset_csv"] == str(chap_project / "input" / "data.csv")
    command(**arguments)
    assert "passed the sanity check" in capsys.readouterr().out


def test_sanity_check_model_applies_a_column_mapping(tmp_path, monkeypatch, data_path, capsys):
    from chap_core.cli_endpoints.utils import sanity_check_model

    monkeypatch.chdir(tmp_path)
    sanity_check_model(
        "builtin:persistence",
        dataset_csv=str(data_path / "laos_subset_custom_columns.csv"),
        data_source_mapping=data_path / "column_mapping.json",
    )
    assert "passed the sanity check" in capsys.readouterr().out


def test_sanity_check_model_keeps_its_old_option_names(tmp_path, monkeypatch, laos_csv):
    monkeypatch.chdir(tmp_path)
    _, arguments = parse(
        "sanity-check-model",
        "--model-url",
        "builtin:persistence",
        "--dataset-path",
        str(laos_csv),
        "--use-local-environement",
    )
    assert arguments == {
        "model_name": "builtin:persistence",
        "dataset_csv": str(laos_csv),
        "use_local_environment": True,
    }


def test_eval_with_validate_refuses_model_options_the_model_does_not_have(tmp_path, models_path, laos_csv):
    from chap_core.cli_endpoints.evaluate import eval_cmd

    model_config = tmp_path / "options.yaml"
    model_config.write_text("user_option_values:\n  some_optoin: '5'\n")
    with pytest.raises(ValueError, match="some_optoin. Its options: some_option"):
        eval_cmd(
            model_name=str(models_path / "naive_python_model_with_mlproject_file_and_docker"),
            dataset_csv=str(laos_csv),
            output_file=tmp_path / "eval.nc",
            model_configuration_yaml=model_config,
            run_config=RunConfig(runs_dir=tmp_path / "runs"),
            validate=True,
        )


@pytest.fixture
def model_requiring_humidity(tmp_path, models_path):
    model = tmp_path / "model"
    shutil.copytree(models_path / "naive_python_model_with_mlproject_file", model)
    with open(model / "MLproject", "a") as mlproject:
        mlproject.write("required_covariates:\n  - mean_relative_humidity\n")
    return model


def test_eval_with_validate_refuses_a_dataset_without_a_covariate_the_model_requires(
    tmp_path, model_requiring_humidity, laos_csv
):
    from chap_core.cli_endpoints.evaluate import eval_cmd

    with pytest.raises(ValueError, match="Required covariate 'mean_relative_humidity' not found"):
        eval_cmd(
            model_name=str(model_requiring_humidity),
            dataset_csv=str(laos_csv),
            output_file=tmp_path / "eval.nc",
            run_config=RunConfig(runs_dir=tmp_path / "runs", ignore_environment=True),
            validate=True,
        )


def test_eval_without_validate_warns_about_a_missing_covariate_and_continues(
    tmp_path, model_requiring_humidity, laos_csv, caplog
):
    from chap_core.api_types import BacktestParams
    from chap_core.cli_endpoints.evaluate import eval_cmd

    with caplog.at_level(logging.WARNING):
        eval_cmd(
            model_name=str(model_requiring_humidity),
            dataset_csv=str(laos_csv),
            output_file=tmp_path / "eval.nc",
            backtest_params=BacktestParams(n_periods=3, n_splits=2),
            run_config=RunConfig(runs_dir=tmp_path / "runs", ignore_environment=True),
        )
    assert (tmp_path / "eval.nc").is_file()
    assert "Required covariate 'mean_relative_humidity' not found" in caplog.text
    assert "use --validate to stop" in caplog.text


def test_validate_reports_a_masked_range_in_the_projects_evaluation_window(chap_project, monkeypatch, capsys):
    (chap_project / "input" / "data_settings.yaml").write_text(
        'maskedRanges:\n  - start: "2012-11"\n    end: "2012-11"\n'
    )
    monkeypatch.chdir(chap_project)
    command, arguments = parse("validate")
    assert arguments["backtest_params"].n_splits == 2
    with pytest.raises(SystemExit):
        command(**arguments)
    assert "Masked period 201211 falls in the evaluation window" in capsys.readouterr().out


def test_plot_dataset_saves_the_plot_to_a_file(tmp_path, laos_csv):
    from chap_core.cli_endpoints.utils import plot_dataset

    plot_dataset(laos_csv, output_file=tmp_path / "plot.html")
    assert "vega" in (tmp_path / "plot.html").read_text()


def test_plot_dataset_without_a_browser_prints_its_url(laos_csv, capsys):
    from http.server import HTTPServer

    from chap_core.cli_endpoints.utils import plot_dataset

    with (
        patch.object(HTTPServer, "serve_forever", side_effect=KeyboardInterrupt),
        patch("webbrowser.open") as open_browser,
    ):
        plot_dataset(laos_csv, browser=False)
    assert "Plot at http://127.0.0.1:" in capsys.readouterr().out
    open_browser.assert_not_called()


def test_init_without_a_directory_asks_before_using_the_current_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    with pytest.raises(ValueError, match="Cancelled"):
        init_cmd(model="builtin:persistence")
    assert not (tmp_path / "chap.yaml").exists()

    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    init_cmd(model="builtin:persistence")
    assert (tmp_path / "chap.yaml").is_file()


def test_init_without_a_directory_or_a_terminal_needs_yes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(ValueError, match="--yes"):
        init_cmd(model="builtin:persistence")
    assert not (tmp_path / "chap.yaml").exists()
    _, arguments = parse("init", "--model", "builtin:persistence", "-y")
    init_cmd(**arguments)
    assert (tmp_path / "chap.yaml").is_file()

"""The ``chap init`` command, which creates a Chap project directory."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from chap_core.external.model_configuration import ModelTemplateConfigV2

from cyclopts import Parameter

from chap_core.api_types import BacktestParams, RunConfig
from chap_core.cli_endpoints._args import (  # noqa: TC001 — used at runtime via cyclopts get_type_hints()
    BacktestParamsArg,
    RunConfigArg,
)
from chap_core.cli_endpoints.model import example_configuration_yaml
from chap_core.project import (
    COLUMN_MAPPING_FILE,
    DATA_SETTINGS_FILE,
    DATASET_FILE,
    INPUT_DIR,
    MODEL_CONFIG_FILE,
    PROJECT_FILE,
    STATE_DIR,
    Project,
)

logger = logging.getLogger(__name__)

PROJECT_FILE_HEADER = f"""\
# Chap project. Run chap commands in this directory.
# Every key is a command-line option: `--backtest-params.n-periods 3` is `backtest_params: {{n_periods: 3}}`.
# An option given on the command line overrides this file. Relative paths are relative to this directory.
# A section named after a command, such as `eval:`, applies to that command only.
# These files in input/ are used when they exist:
#   {DATASET_FILE} (--dataset-csv), {DATA_SETTINGS_FILE} (--data-configuration),
#   {COLUMN_MAPPING_FILE} (--data-source-mapping), {MODEL_CONFIG_FILE} (--model-configuration-yaml)
"""

# Written to input/data_settings.yaml so the file and its format are visible; with only comments it sets nothing.
DATA_SETTINGS_TEMPLATE = """\
# Data settings for input/data.csv, used as --data-configuration.
# Masked ranges hide disease_cases in periods you do not trust, such as a reporting gap,
# from training and scoring. Remove the # signs to use them:
#
# maskedRanges:
#   - start: "2021-06"
#     end: "2021-08"
"""


def init_cmd(
    directory: Annotated[
        Path | None,
        Parameter(
            help="Project directory. Created if it does not exist. Without it, the current directory becomes "
            "the project, after you confirm."
        ),
    ] = None,
    *,
    model: Annotated[
        str | None,
        Parameter(
            help="Model to evaluate: a Chap Hub model ID (stored as its verified stable image), local directory, "
            "GitHub URL, chapkit service URL, chapkit container image (docker://<image>) or built-in model "
            "(builtin:<name>). Can also be set later as model_name in chap.yaml."
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        Parameter(help="Path or URL to the dataset CSV. Copied to input/data.csv, with its .geojson if there is one."),
    ] = None,
    data_settings: Annotated[
        Path | None,
        Parameter(help="Data settings file, as for --data-configuration. Copied to input/data_settings.yaml."),
    ] = None,
    column_mapping: Annotated[
        Path | None,
        Parameter(help="Column mapping file, as for --data-source-mapping. Copied to input/column_mapping.json."),
    ] = None,
    model_config: Annotated[
        Path | None,
        Parameter(
            help="Model configuration file, as for --model-configuration-yaml. Copied to input/model_config.yaml. "
            "Without it, the file is written with the model's default option values."
        ),
    ] = None,
    backtest_params: BacktestParamsArg = BacktestParams(),
    run_config: RunConfigArg = RunConfig(),
    pin: Annotated[
        bool,
        Parameter(
            help="Pin a GitHub model to its current commit, so every run uses the same model code. "
            "With --no-pin, each run uses the latest commit."
        ),
    ] = True,
    force: Annotated[bool, Parameter(help="Overwrite an existing chap.yaml.")] = False,
    yes: Annotated[
        bool,
        Parameter(
            alias="-y",
            negative="",
            help="Make the current directory the project without asking, when no directory is given.",
        ),
    ] = False,
) -> None:
    """Create a Chap project, so that `chap eval` runs without options inside it.

    Writes chap.yaml with the model and evaluation settings, and creates input/ for the dataset and
    configuration files, output/ for results and .chap/ for model working copies.

    Examples:
        chap init laos --model builtin:seasonal_median --dataset ./example_data/laos_subset.csv
        cd laos
        chap eval
    """
    import yaml

    from chap_core.cli_endpoints._common import discover_geojson, load_data_configuration, resolve_csv_path
    from chap_core.log_config import initialize_logging

    initialize_logging()

    if directory is None:
        directory = Path()
        if not yes:
            _confirm_current_directory()
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"{directory} is a file, not a directory")
    if (directory / PROJECT_FILE).exists() and not force:
        raise ValueError(f"{directory / PROJECT_FILE} already exists. Use --force to overwrite it.")
    for path in (data_settings, column_mapping, model_config):
        if path is not None and not path.is_file():
            raise ValueError(f"{path} is not a file")
    if data_settings is not None:
        load_data_configuration(data_settings)
    csv_path, geojson_path = resolve_csv_path(dataset) if dataset is not None else (None, None)
    if csv_path is not None and not csv_path.is_file():
        raise ValueError(f"{dataset} is not a file")
    input_dir = directory / INPUT_DIR
    _check_dataset(
        csv_path or input_dir / DATASET_FILE,
        column_mapping or input_dir / COLUMN_MAPPING_FILE,
        data_settings or input_dir / DATA_SETTINGS_FILE,
        changed=(dataset, column_mapping, data_settings),
    )
    # Loading the model checks that it can be found before anything is written.
    template_config = None
    if model is not None:
        model = resolve_model_source(model, pin)
        template_config = _model_template_config(model, run_config)
    if model_config is not None:
        _check_model_config(model_config, template_config)

    previous_model = _previous_model(directory)
    directory.mkdir(parents=True, exist_ok=True)
    project = Project(root=directory.resolve(), settings={})
    project_model = _relative_to_project(model, project) if model is not None else None
    for path in (project.input_dir, project.output_dir, project.runs_dir):
        path.mkdir(parents=True, exist_ok=True)

    created: list[str] = []
    if csv_path is not None:
        shutil.copyfile(csv_path, project.input_dir / DATASET_FILE)
        created.append(f"input/{DATASET_FILE}")
        geojson_path = geojson_path or discover_geojson(csv_path)
        if geojson_path is not None:
            shutil.copyfile(geojson_path, project.input_dir / "data.geojson")
            created.append("input/data.geojson")

    data_settings_path = project.input_dir / DATA_SETTINGS_FILE
    if data_settings is not None:
        content = yaml.safe_load(data_settings.read_text(encoding="utf-8"))
        data_settings_path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
        created.append(f"input/{DATA_SETTINGS_FILE}")
    elif not data_settings_path.exists():
        data_settings_path.write_text(DATA_SETTINGS_TEMPLATE, encoding="utf-8")
        created.append(f"input/{DATA_SETTINGS_FILE} (example, no settings yet)")
    if column_mapping is not None:
        shutil.copyfile(column_mapping, project.input_dir / COLUMN_MAPPING_FILE)
        created.append(f"input/{COLUMN_MAPPING_FILE}")
    model_config_path = project.input_dir / MODEL_CONFIG_FILE
    if model_config is not None:
        shutil.copyfile(model_config, model_config_path)
        created.append(f"input/{MODEL_CONFIG_FILE}")
    elif template_config is not None and project_model is not None:
        options = template_config.user_options or {}
        if project_model == previous_model and model_config_path.exists():
            # Re-initialising with the same model keeps the options you edited.
            created.append(f"input/{MODEL_CONFIG_FILE} (kept, same model)")
        elif options:
            model_config_path.write_text(example_configuration_yaml(template_config, project_model), encoding="utf-8")
            missing = [name for name, spec in options.items() if spec.get("default") is None]
            created.append(f"input/{MODEL_CONFIG_FILE}" + (f" (fill in {', '.join(missing)})" if missing else ""))
        else:
            # Options written for another model do not apply to this one, which has none.
            model_config_path.unlink(missing_ok=True)

    (project.root / PROJECT_FILE).write_text(
        _project_file(project_model, backtest_params, run_config), encoding="utf-8"
    )
    _ignore_state_dir(project.root / ".gitignore")

    print(f"Created Chap project in {directory}")
    if project_model is not None:
        print(f"  model: {project_model}")
    for name in [PROJECT_FILE, *created]:
        print(f"  {name}")
    todo = [f"set model_name in {PROJECT_FILE}"] if model is None else []
    todo += [f"copy your dataset to input/{DATASET_FILE}"] if not (project.input_dir / DATASET_FILE).is_file() else []
    # Project commands only use the chap.yaml in the current directory.
    cd = f"`cd {directory}`, then " if project.root != Path.cwd().resolve() else ""
    if todo:
        print(f"Next: {cd}{' and '.join(todo)}, then run `chap eval`.")
    else:
        print(f"Next: {cd}`chap validate` to check the dataset against the model, then `chap eval`.")


def _confirm_current_directory() -> None:
    """Ask before turning the current directory into a project; raise ValueError if the answer is no."""
    import sys

    cwd = Path.cwd()
    if not sys.stdin.isatty():
        raise ValueError(
            f"chap init without a directory makes the current directory ({cwd}) a Chap project. "
            "Give a directory name, or --yes to use the current directory."
        )
    answer = input(
        f"Make the current directory {cwd} a Chap project? It writes chap.yaml, input/ and output/ here. [y/N] "
    )
    if answer.strip().lower() not in ("y", "yes"):
        raise ValueError("Cancelled; nothing was written. Give a directory name, such as `chap init my-project`.")


def _previous_model(directory: Path) -> str | None:
    """The model_name in an existing chap.yaml that chap init --force replaces, or None."""
    if not (directory / PROJECT_FILE).is_file():
        return None
    try:
        model = Project.load(directory).settings.get("model_name")
    except ValueError:
        return None
    return model if isinstance(model, str) else None


def resolve_model_source(model: str, pin: bool) -> str:
    """The model source to store in chap.yaml for the ``--model`` given to chap init.

    A Chap Hub model ID becomes ``docker://`` with its verified stable image. With ``pin``, a GitHub URL
    without a commit gets the commit its default branch points at now. Other sources are kept as given.
    """
    from chap_core.models.chapkit_service_manager import DOCKER_SOURCE_PREFIX

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", model) and not Path(model).exists():
        from chap_core.services.model_marketplace import resolve_model

        try:
            hub_pin = resolve_model(model)
        except ValueError as e:
            raise ValueError(f"{model!r} is not a local directory, and not a Chap Hub model: {e}") from e
        logger.info(f"Chap Hub model {model} {hub_pin.version} is {hub_pin.image}")
        return f"{DOCKER_SOURCE_PREFIX}{hub_pin.image}"
    if pin and model.startswith("https://github.com/") and "@" not in model:
        url = model.rstrip("/")
        commit = remote_head_commit(url)
        logger.info(f"Pinned {url} to commit {commit}; use --no-pin to follow the latest commit instead")
        return f"{url}@{commit}"
    return model


def remote_head_commit(repository: str) -> str:
    """The commit a git repository's default branch points at, read with ``git ls-remote``."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", repository, "HEAD"], capture_output=True, text=True, check=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise ValueError(f"Could not read the current commit of {repository}: {e}") from e
    commit = result.stdout.split()[0] if result.stdout.split() else ""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"Could not read the current commit of {repository}: {result.stdout.strip()!r}")
    return commit


def _model_template_config(model: str, run_config: RunConfig) -> ModelTemplateConfigV2:
    """Load the model once to read its configuration, including its options."""
    from chap_core.cli_endpoints._common import inspect_model_template

    with inspect_model_template(model, run_config) as template:
        config: ModelTemplateConfigV2 = template.model_template_config
        return config


def _check_dataset(
    csv_path: Path, column_mapping: Path, data_settings: Path, changed: tuple[object, object, object]
) -> None:
    """Load the dataset the project will use, with its column mapping and data settings, before writing anything.

    Only runs when chap init is given one of them (``changed``); a file not given is the one already in
    input/, so a kept data_settings.yaml that does not fit a new dataset is caught here.
    """
    import json

    from chap_core.cli_endpoints._common import load_data_configuration, load_dataset_from_csv

    if not any(value is not None for value in changed) or not csv_path.is_file():
        if changed[1] is not None:
            _read_column_mapping(column_mapping)
        return
    used = [path for path in (column_mapping, data_settings) if path.is_file()]
    try:
        mapping = _read_column_mapping(column_mapping) if column_mapping.is_file() else None
        settings = load_data_configuration(data_settings) if data_settings.is_file() else None
        load_dataset_from_csv(csv_path, column_mapping=mapping, data_configuration=settings)
    except (ValueError, TypeError, KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        with_files = f" with {' and '.join(str(path) for path in used)}" if used else ""
        raise ValueError(
            f"Could not load {csv_path} as a Chap dataset{with_files}: {e}\n"
            f"Run `chap validate --dataset-csv {csv_path}` for a full report."
        ) from e


def _read_column_mapping(path: Path) -> dict[str, str]:
    import json

    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Could not read the column mapping {path}: {e}") from e
    if not isinstance(mapping, dict):
        raise ValueError(f"The column mapping {path} must be a JSON object of covariate names to column names")
    return mapping


def _check_model_config(path: Path, template_config: ModelTemplateConfigV2 | None) -> None:
    """Check that a model configuration file can be read and, when the model is known, only sets its options."""
    import yaml
    from pydantic import ValidationError

    from chap_core.cli_endpoints._common import check_model_configuration
    from chap_core.database.model_templates_and_config_tables import ModelConfiguration

    try:
        configuration = ModelConfiguration.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except (yaml.YAMLError, ValidationError) as e:
        raise ValueError(f"Could not read the model configuration {path}: {e}") from e
    if template_config is not None:
        check_model_configuration(configuration, template_config, path)


def _relative_to_project(model: str, project: Project) -> str:
    """A local model directory inside or next to the project as a path relative to it, since chap.yaml paths are.

    A model directory further away is stored as an absolute path, and other sources as given.
    """
    import os

    path = Path(model).expanduser()
    if "://" in model or not path.is_dir():
        return model
    try:
        relative = Path(os.path.relpath(path.resolve(), project.root))
    except ValueError:  # on another drive than the project, on Windows
        return str(path.resolve())
    # Only a model inside the project or next to it moves with the project; one further away keeps its place.
    if relative.parts[:2] == ("..", ".."):
        return str(path.resolve())
    return relative.as_posix()


def _project_file(model: str | None, backtest_params: BacktestParams, run_config: RunConfig) -> str:
    import yaml

    model_line = (
        yaml.safe_dump({"model_name": model})
        if model is not None
        else "# model_name: <local directory, GitHub URL, chapkit service URL, docker://<image> or builtin:<name>>\n"
    )
    # Only the run options given to chap init, such as is_chapkit_model, so the rest keep their defaults.
    run_options = run_config.model_dump(mode="json", exclude_defaults=True)
    run_lines = yaml.safe_dump({"run_config": run_options}, sort_keys=False) if run_options else ""
    evaluation = {"eval": {"backtest_params": backtest_params.model_dump(mode="json"), "plot": True}}
    return f"{PROJECT_FILE_HEADER}\n{model_line}{run_lines}\n{yaml.safe_dump(evaluation, sort_keys=False)}"


def _ignore_state_dir(gitignore: Path) -> None:
    """Add .chap/ to the project's .gitignore: it holds model working copies, not project files."""
    entry = f"{STATE_DIR}/"
    lines = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    if entry not in lines:
        gitignore.write_text("\n".join([*lines, entry]) + "\n", encoding="utf-8")


def register_commands(app):
    """Register the init command with the CLI app."""
    app.command(name="init")(init_cmd)

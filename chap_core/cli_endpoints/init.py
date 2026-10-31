"""The ``chap init`` command, which creates a Chap project directory."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from chap_core.api_types import BacktestParams
from chap_core.cli_endpoints._args import BacktestParamsArg  # noqa: TC001 — used at runtime via cyclopts get_type_hints()
from chap_core.project import (
    COLUMN_MAPPING_FILE,
    DATA_SETTINGS_FILE,
    DATASET_FILE,
    MODEL_CONFIG_FILE,
    PROJECT_FILE,
    STATE_DIR,
    Project,
)

logger = logging.getLogger(__name__)

PROJECT_FILE_HEADER = f"""\
# Chap project. Run chap commands in this directory or in any directory below it.
# Every key is a command-line option: `--backtest-params.n-periods 3` is `backtest_params: {{n_periods: 3}}`.
# An option given on the command line overrides this file. Relative paths are relative to this directory.
# A section named after a command, such as `eval:`, applies to that command only.
# These files in input/ are used when they exist:
#   {DATASET_FILE} (--dataset-csv), {DATA_SETTINGS_FILE} (--data-configuration),
#   {COLUMN_MAPPING_FILE} (--data-source-mapping), {MODEL_CONFIG_FILE} (--model-configuration-yaml)
"""


def init_cmd(
    directory: Annotated[Path, Parameter(help="Project directory. Created if it does not exist.")] = Path(),
    *,
    model: Annotated[
        str | None,
        Parameter(
            help="Model to evaluate: local directory, GitHub URL, chapkit service URL or built-in model "
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
    force: Annotated[bool, Parameter(help="Overwrite an existing chap.yaml.")] = False,
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

    if (directory / PROJECT_FILE).exists() and not force:
        raise ValueError(f"{directory / PROJECT_FILE} already exists. Use --force to overwrite it.")
    for path in (data_settings, column_mapping, model_config):
        if path is not None and not path.is_file():
            raise ValueError(f"{path} does not exist")
    if data_settings is not None:
        load_data_configuration(data_settings)

    directory.mkdir(parents=True, exist_ok=True)
    project = Project(root=directory.resolve(), settings={})
    for path in (project.input_dir, project.output_dir, project.runs_dir):
        path.mkdir(parents=True, exist_ok=True)

    created: list[str] = []
    if dataset is not None:
        csv_path, geojson_path = resolve_csv_path(dataset)
        if not csv_path.is_file():
            raise ValueError(f"{dataset} does not exist")
        shutil.copyfile(csv_path, project.input_dir / DATASET_FILE)
        created.append(f"input/{DATASET_FILE}")
        geojson_path = geojson_path or discover_geojson(csv_path)
        if geojson_path is not None:
            shutil.copyfile(geojson_path, project.input_dir / "data.geojson")
            created.append("input/data.geojson")

    if data_settings is not None:
        content = yaml.safe_load(data_settings.read_text(encoding="utf-8"))
        (project.input_dir / DATA_SETTINGS_FILE).write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
        created.append(f"input/{DATA_SETTINGS_FILE}")
    if column_mapping is not None:
        shutil.copyfile(column_mapping, project.input_dir / COLUMN_MAPPING_FILE)
        created.append(f"input/{COLUMN_MAPPING_FILE}")
    if model_config is not None:
        shutil.copyfile(model_config, project.input_dir / MODEL_CONFIG_FILE)
        created.append(f"input/{MODEL_CONFIG_FILE}")
    elif model is not None:
        missing = _write_default_model_config(model, project)
        if (project.input_dir / MODEL_CONFIG_FILE).exists():
            note = f" (fill in {', '.join(missing)})" if missing else ""
            created.append(f"input/{MODEL_CONFIG_FILE}{note}")

    (project.root / PROJECT_FILE).write_text(_project_file(model, backtest_params), encoding="utf-8")
    _ignore_state_dir(project.root / ".gitignore")

    print(f"Created Chap project in {directory}")
    for name in [PROJECT_FILE, *created]:
        print(f"  {name}")
    if model is None:
        print(f"Set model_name in {PROJECT_FILE}, then run `chap eval`.")
    elif dataset is None:
        print(f"Copy your dataset to input/{DATASET_FILE}, then run `chap eval`.")
    else:
        print("Next: `chap validate` to check the dataset against the model, then `chap eval`.")


def _write_default_model_config(model: str, project: Project) -> list[str]:
    """Write input/model_config.yaml with the model's default option values, if the model has options.

    Loading the model also checks that it can be found. Returns the options that have no default.
    """
    import yaml

    from chap_core.cli_endpoints.model import build_example_configuration
    from chap_core.models.model_template import ModelTemplate

    logger.info(f"Loading model template from {model}")
    template = ModelTemplate.from_directory_or_github_url(
        model, base_working_dir=project.runs_dir, ignore_env=True, dry_run=True
    )
    with template:
        example = build_example_configuration(template.model_template_config)
    options = example["user_option_values"]
    if not options:
        return []
    (project.input_dir / MODEL_CONFIG_FILE).write_text(yaml.safe_dump(example, sort_keys=False), encoding="utf-8")
    return [name for name, value in options.items() if value is None]


def _project_file(model: str | None, backtest_params: BacktestParams) -> str:
    import yaml

    model_line = (
        yaml.safe_dump({"model_name": model})
        if model is not None
        else "# model_name: <local directory, GitHub URL, chapkit service URL or builtin:<name>>\n"
    )
    evaluation = {"eval": {"backtest_params": backtest_params.model_dump(mode="json"), "plot": True}}
    return f"{PROJECT_FILE_HEADER}\n{model_line}\n{yaml.safe_dump(evaluation, sort_keys=False)}"


def _ignore_state_dir(gitignore: Path) -> None:
    """Add .chap/ to the project's .gitignore: it holds model working copies, not project files."""
    entry = f"{STATE_DIR}/"
    lines = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    if entry not in lines:
        gitignore.write_text("\n".join([*lines, entry]) + "\n", encoding="utf-8")


def register_commands(app):
    """Register the init command with the CLI app."""
    app.command(name="init")(init_cmd)

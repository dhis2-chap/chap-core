"""Chap projects: a directory with a ``chap.yaml`` file that gives chap commands their defaults.

``chap init`` creates a project. Inside it, the commands in ``PROJECT_COMMANDS`` fill every
option that is not given on the command line from, in order:

1. the section of ``chap.yaml`` named after the command, such as ``eval:``
2. the top-level keys of ``chap.yaml``
3. the files with fixed names in ``input/``, such as ``input/data.csv``
4. project defaults: a new file in ``output/``, and ``.chap/runs`` for model working copies

Every key in ``chap.yaml`` is a command-line option: ``--backtest-params.n-periods 3`` is
``backtest_params: {n_periods: 3}``. Relative paths in it are relative to the project root.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

if TYPE_CHECKING:
    from cyclopts import App
    from cyclopts.argument import Argument, ArgumentCollection

logger = logging.getLogger(__name__)

PROJECT_FILE = "chap.yaml"
INPUT_DIR = "input"
OUTPUT_DIR = "output"
STATE_DIR = ".chap"

DATASET_FILE = "data.csv"
DATA_SETTINGS_FILE = "data_settings.yaml"
COLUMN_MAPPING_FILE = "column_mapping.json"
MODEL_CONFIG_FILE = "model_config.yaml"

# Commands that take their defaults from a project.
PROJECT_COMMANDS = {("eval",), ("validate",), ("model", "schema")}
SECTION_NAMES = {command[0] for command in PROJECT_COMMANDS}

# Each fixed-name file in input/, the option it fills, and the option it belongs to. A file is
# not used when the option it belongs to is given on the command line, so that, for example,
# masked ranges written for the project's dataset are not applied to another dataset.
INPUT_FILES = (
    (DATASET_FILE, "dataset_csv", None),
    (DATA_SETTINGS_FILE, "data_configuration", "dataset_csv"),
    (COLUMN_MAPPING_FILE, "data_source_mapping", "dataset_csv"),
    (MODEL_CONFIG_FILE, "model_configuration_yaml", "model_name"),
)

# Options typed as plain strings that still hold a local path unless they are a URL or built-in model.
STRING_PATH_OPTIONS = {"model_name", "dataset_csv"}


@dataclass(frozen=True)
class Project:
    """A Chap project: its root directory and the settings read from its ``chap.yaml``."""

    root: Path
    settings: dict[str, Any]

    @property
    def input_dir(self) -> Path:
        return self.root / INPUT_DIR

    @property
    def output_dir(self) -> Path:
        return self.root / OUTPUT_DIR

    @property
    def runs_dir(self) -> Path:
        """Where model working copies go, instead of ``runs/`` in the current directory."""
        return self.root / STATE_DIR / "runs"

    @classmethod
    def load(cls, root: Path) -> Project:
        """Read ``chap.yaml`` in ``root``. Raises ValueError when it cannot be read or is not a mapping."""
        import yaml

        path = root / PROJECT_FILE
        try:
            settings = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as e:
            raise ValueError(f"Could not read {path}: {e}") from e
        if settings is None:
            settings = {}
        if not isinstance(settings, dict):
            raise ValueError(f"{path} must be a mapping of option names to values")
        return cls(root=root.resolve(), settings=settings)


def find_project(start: Path | None = None) -> Project | None:
    """Return the project whose ``chap.yaml`` is in ``start`` (default: the current directory) or above it."""
    directory = (start or Path.cwd()).resolve()
    for candidate in (directory, *directory.parents):
        if (candidate / PROJECT_FILE).is_file():
            return Project.load(candidate)
    return None


def model_label(source: str) -> str:
    """A short, file-name-safe name for a model source, used to name output files.

    ``builtin:seasonal_median@1`` gives ``seasonal_median``, ``https://github.com/org/repo@abc`` gives
    ``repo``, ``./models/my_model`` gives ``my_model`` and ``http://localhost:8000`` gives ``localhost-8000``.
    """
    name = source.removeprefix("builtin:").split("@")[0].rstrip("/")
    name = name.rsplit("/", 1)[-1].removesuffix(".git")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "model"


def project_defaults(app: App, commands: tuple[str, ...], arguments: ArgumentCollection) -> None:
    """Cyclopts config hook that fills options not given on the command line from the current project.

    Does nothing outside a project or for commands not in ``PROJECT_COMMANDS``.
    """
    commands = tuple(commands)
    if commands not in PROJECT_COMMANDS:
        return
    project = find_project()
    if project is None:
        return

    from cyclopts.config import Dict

    values, sources = project_values(project, commands, arguments)
    used = [f"{option} ({sources[option]})" for option in values if _is_used(arguments, option)]
    logger.info("Using Chap project at %s: %s", project.root, ", ".join(used) or "no values")
    Dict(values, use_commands_as_keys=False, allow_unknown=True, source=str(project.root / PROJECT_FILE))(
        app, commands, arguments
    )


def project_values(
    project: Project, commands: tuple[str, ...], arguments: ArgumentCollection
) -> tuple[dict[str, Any], dict[str, str]]:
    """The option values a project gives a command, and where each top-level option came from."""
    settings = project.settings
    shared = {key: value for key, value in settings.items() if key not in SECTION_NAMES}
    section: Any = settings
    for command in commands:
        section = section.get(command) if isinstance(section, dict) else None
    if section is not None and not isinstance(section, dict):
        raise ValueError(f"The {' '.join(commands)!r} section of {project.root / PROJECT_FILE} must be a mapping")

    _check_options(shared, section or {}, commands, arguments)
    values = _resolve_paths(_merge(shared, section or {}), project.root, arguments)
    sources = dict.fromkeys(values, PROJECT_FILE)

    for filename, option, owner in INPUT_FILES:
        path = project.input_dir / filename
        if option in values or not path.is_file() or (owner is not None and _given(arguments, owner)):
            continue
        values[option] = str(path)
        sources[option] = f"{INPUT_DIR}/{filename}"

    if commands == ("eval",):
        if "output_file" not in values:
            model = _given(arguments, "model_name") or str(values.get("model_name", "model"))
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            values["output_file"] = str(project.output_dir / f"{model_label(model)}-{stamp}.nc")
            sources["output_file"] = f"{OUTPUT_DIR}/"
        run_config = values.get("run_config")
        if run_config is None or isinstance(run_config, dict):
            values["run_config"] = {"runs_dir": str(project.runs_dir), **(run_config or {})}
            sources.setdefault("run_config", f"{STATE_DIR}/")
    return values, sources


def _check_options(
    shared: dict[str, Any], section: dict[str, Any], commands: tuple[str, ...], arguments: ArgumentCollection
) -> None:
    """Catch misspelled keys in ``chap.yaml``, which would otherwise leave an option at its default.

    Raises ValueError for a key the command should take but does not: any key in its own section, and
    any nested key under a top-level option it takes. Warns about top-level keys no project command takes.
    """
    unknown_shared = []
    for key, value in shared.items():
        if _match(arguments, (key,)) is not None:
            _check_leaves({key: value}, commands, arguments)
        elif key not in project_options():
            unknown_shared.append(key)
    _check_leaves(section, commands, arguments, section=commands)
    if unknown_shared:
        logger.warning(
            "%s has keys that no project command (%s) takes: %s",
            PROJECT_FILE,
            ", ".join(" ".join(command) for command in sorted(PROJECT_COMMANDS)),
            ", ".join(unknown_shared),
        )


def _check_leaves(
    values: dict[str, Any],
    commands: tuple[str, ...],
    arguments: ArgumentCollection,
    keys: tuple[str, ...] = (),
    section: tuple[str, ...] = (),
) -> None:
    for key, value in values.items():
        path = (*keys, key)
        if isinstance(value, dict):
            _check_leaves(value, commands, arguments, path, section)
        elif _match(arguments, path) is None:
            where = f"the {'.'.join(section)} section of " if section else ""
            raise ValueError(
                f"Unknown option {'.'.join(path)!r} in {where}{PROJECT_FILE} for chap {' '.join(commands)}"
            )


def project_options() -> set[str]:
    """Names of the top-level options of all commands in ``PROJECT_COMMANDS``."""
    import inspect

    from chap_core.cli_endpoints.evaluate import eval_cmd
    from chap_core.cli_endpoints.model import schema_cmd
    from chap_core.cli_endpoints.validate import validate_cmd

    return {name for command in (eval_cmd, validate_cmd, schema_cmd) for name in inspect.signature(command).parameters}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base``, key by key for nested mappings."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_paths(
    values: dict[str, Any], root: Path, arguments: ArgumentCollection, keys: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Make the relative paths in ``values`` relative to ``root`` instead of the current directory."""
    resolved: dict[str, Any] = {}
    for key, value in values.items():
        path = (*keys, key)
        if isinstance(value, dict):
            resolved[key] = _resolve_paths(value, root, arguments, path)
        elif isinstance(value, str) and _is_path_option(_match(arguments, path), key, value):
            resolved[key] = str(root / Path(value).expanduser())
        else:
            resolved[key] = value
    return resolved


def _is_path_option(argument: Argument | None, key: str, value: str) -> bool:
    if argument is None or "://" in value or value.startswith("builtin:"):
        return False
    return key in STRING_PATH_OPTIONS or Path in (argument.hint, *get_args(argument.hint))


def _match(arguments: ArgumentCollection, keys: tuple[str, ...]) -> Argument | None:
    try:
        argument, _, _ = arguments.match("--" + ".".join(key.replace("_", "-") for key in keys))
    except ValueError:
        return None
    return argument


def _given(arguments: ArgumentCollection, option: str) -> str | None:
    """The value of ``option`` given on the command line, or None."""
    argument = _match(arguments, (option,))
    if argument is None or not argument.tokens:
        return None
    return argument.tokens[0].value


def _is_used(arguments: ArgumentCollection, option: str) -> bool:
    """Whether a project value for ``option`` applies: the command takes it and it is not on the command line."""
    argument = _match(arguments, (option,))
    return argument is not None and not argument.tokens

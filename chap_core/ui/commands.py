"""Describe chap CLI commands as form fields and turn filled-in values back into CLI arguments.

Fields are read from the same cyclopts definitions the CLI parses, so every command and
option is available in the UI and stays in sync with the CLI.
"""

from __future__ import annotations

import enum
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Parameters whose value is a file the command writes. Relative values are resolved inside the run directory.
OUTPUT_PARAMS = {"output_file", "out_file", "out_path", "output_prefix", "report_filename", "state_file", "output_csv"}

# UI-only defaults, keyed by field key: output names so the user does not have to invent them,
# and values that make a command usable without a terminal.
UI_DEFAULTS: dict[str, dict[str, Any]] = {
    "eval": {"output_file": "evaluation.nc"},
    "evaluate-ensemble": {"report_filename": "ensemble_report.csv", "output_file": "evaluation.nc"},
    "causal": {"output_file": "causal.nc"},
    "causal build-counterfactual": {"output_csv": "counterfactual.csv"},
    "aggregate-eval": {"output_file": "aggregated.nc"},
    "model schema": {"output_file": "model_schema.yaml"},
    "report": {"out_file": "report.pdf"},
    "generate-modelcard": {"output_file": "modelcard.md"},
    "plot-backtest": {"output_file": "plot.html"},
    "generate-pdf-report": {"output_file": "report.pdf"},
    "export-metrics": {"output_file": "metrics.csv"},
    "write-open-api-spec": {"out_path": "openapi.json"},
    "convert-request": {"output_prefix": "request"},
    "forecast": {"out_path": "./"},
    "multi-forecast": {"out_path": "./"},
    "preference-learn": {"state_file": "preference_state.json", "learning_params.decision_mode": "metric"},
}

# Friendlier labels than the ones derived from flag names.
LABELS = {
    "model_name": "Model",
    "model_url": "Model",
    "dataset_csv": "Dataset",
    "backtest_params.n_periods": "Periods to forecast",
    "backtest_params.n_splits": "Test splits",
    "backtest_params.stride": "Periods between splits",
    "backtest_params.n_retrain": "Times to retrain",
    "model_configuration_yaml": "Model configuration",
    "model_config_path": "Model configuration",
    "data_source_mapping": "Column mapping (JSON)",
    "estimator_options.max_trials": "Max trials",
    "input_file": "Evaluation file",
    "input_files": "Evaluation files",
    "evaluation_path": "Evaluation file",
}

# Free-text CLI options that are really a choice from a registry.
CHOICE_SOURCES: dict[str, str] = {
    "plot_type": "backtest_plots",
    "plot_name": "dataset_plots",
    "metric_ids": "metrics",
}


@dataclass(frozen=True)
class Field:
    """One CLI option, as the UI should render it."""

    flag: str
    key: str
    label: str
    kind: str  # bool | int | float | str | path | choice | multichoice | list
    default: Any  # value the UI starts with
    cli_default: Any  # value the CLI uses when the option is left out
    required: bool
    optional: bool
    help: str
    group: str | None
    choices: tuple[str, ...] = ()
    negative_flag: str | None = None

    @property
    def is_output(self) -> bool:
        return self.key in OUTPUT_PARAMS


def get_command(command: str):
    """The cyclopts sub-app for a (possibly nested, space separated) command name."""
    from chap_core.cli import app

    sub = app
    for part in command.split():
        sub = sub[part]
    return sub


def list_commands() -> list[str]:
    """Every runnable chap command, including nested ones like `model schema`."""
    from chap_core.cli import app

    commands = []
    for name in app:
        if name.startswith("-") or name == "ui":
            continue
        sub = app[name]
        if sub.default_command is not None:
            commands.append(name)
        commands += [f"{name} {child}" for child in sub if not child.startswith("-")]
    return sorted(commands)


def command_help(command: str) -> str:
    """First paragraph of the command's docstring."""
    func = get_command(command).default_command
    doc = (func.__doc__ or "").strip()
    return doc.split("\n\n")[0].replace("\n", " ")


def command_fields(command: str) -> list[Field]:
    """Form fields for every leaf option of a command, in CLI order."""
    arguments = get_command(command).assemble_argument_collection()
    groups = {arg.name: arg.name.removeprefix("--") for arg in arguments if arg.children}
    fields = []
    for arg in arguments:
        if arg.children or not arg.parse:
            continue
        flag = next(name for name in arg.names if name.startswith("--") and not name.startswith("--no-"))
        negative = next((name for name in arg.names if name.startswith("--no-") or ".no-" in name), None)
        parent = flag.rsplit(".", 1)[0] if "." in flag else None
        hint, optional = _unwrap_optional(arg.hint)
        kind, choices = _kind(hint)
        cli_default = arg.field_info.default
        if cli_default is arg.field_info.empty:
            cli_default = None
        if isinstance(cli_default, enum.Enum):
            cli_default = cli_default.value
        if isinstance(cli_default, Path):
            cli_default = str(cli_default)
        key = flag.removeprefix("--").replace("-", "_")
        if key in CHOICE_SOURCES:
            choices = registry_choices(CHOICE_SOURCES[key])
            kind = "multichoice" if kind == "list" else "choice"
        fields.append(
            Field(
                flag=flag,
                key=key,
                label=LABELS.get(key) or flag.rsplit(".", 1)[-1].removeprefix("--").replace("-", " ").capitalize(),
                kind=kind,
                default=UI_DEFAULTS.get(command, {}).get(key, cli_default),
                cli_default=cli_default,
                required=bool(arg.required),
                optional=optional,
                help=arg.parameter.help or "",
                group=groups.get(parent) if parent else None,
                choices=choices,
                negative_flag=negative if kind == "bool" else None,
            )
        )
    return fields


def build_args(command: str, fields: list[Field], values: dict[str, Any]) -> list[str]:
    """CLI arguments for a command, leaving out options that keep their default value."""
    args = command.split()
    for field in fields:
        value = values.get(field.key)
        if value is None or value == "" or value == []:
            continue
        if field.kind == "bool":
            if bool(value) != bool(field.cli_default):
                args.append(field.flag if value else _negative(field))
            continue
        if field.kind in ("list", "multichoice"):
            items = [str(item) for item in value]
            if field.cli_default is not None and items == [str(item) for item in field.cli_default]:
                continue
            for item in items:
                args += [field.flag, item]
            continue
        if not field.required and field.cli_default is not None and str(value) == str(field.cli_default):
            continue
        args += [field.flag, str(value)]
    return args


def registry_choices(source: str) -> tuple[str, ...]:
    """Ids registered for plots or metrics, used as choices for free-text CLI options."""
    if source == "backtest_plots":
        from chap_core.cli_endpoints.generated_plot_ids import BACKTEST_PLOT_IDS

        return tuple(BACKTEST_PLOT_IDS)
    if source == "dataset_plots":
        from chap_core.plotting.dataset_plot import get_dataset_plots_registry

        return tuple(get_dataset_plots_registry())
    from chap_core.assessment.metrics import available_metrics

    return tuple(available_metrics)


def _negative(field: Field) -> str:
    if field.negative_flag:
        return field.negative_flag
    head, _, tail = field.flag.rpartition(".")
    return f"{head}.no-{tail}" if head else f"--no-{field.flag.removeprefix('--')}"


def _unwrap_optional(hint) -> tuple[Any, bool]:
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return hint, False


def _kind(hint) -> tuple[str, tuple[str, ...]]:
    origin = typing.get_origin(hint)
    if origin is typing.Literal:
        return "choice", tuple(str(a) for a in typing.get_args(hint))
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return "choice", tuple(str(member.value) for member in hint)
    if origin in (list, tuple, set) or hint in (list, tuple):
        return "list", ()
    if hint is bool:
        return "bool", ()
    if hint is int:
        return "int", ()
    if hint is float:
        return "float", ()
    if isinstance(hint, type) and issubclass(hint, Path):
        return "path", ()
    return "str", ()

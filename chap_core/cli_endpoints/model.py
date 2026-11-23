"""Model introspection commands for the CHAP CLI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path  # noqa: TC003 — used at runtime via cyclopts get_type_hints()
from typing import TYPE_CHECKING, Annotated, Any

from cyclopts import Parameter

from chap_core.api_types import RunConfig
from chap_core.cli_endpoints._args import (  # noqa: TC001 — used at runtime via cyclopts get_type_hints()
    ModelNameArg,
    RunConfigArg,
)

if TYPE_CHECKING:
    from chap_core.external.model_configuration import ModelTemplateConfigV2

logger = logging.getLogger(__name__)


def build_model_configuration_schema(template_config: ModelTemplateConfigV2) -> dict[str, Any]:
    """Build a JSON schema describing the model_configuration_yaml input for the given template.

    Wraps the template's ``user_options`` inside a ``user_option_values`` object and pairs it
    with ``additional_continuous_covariates`` so the schema mirrors the on-disk YAML shape
    accepted by ``--model-configuration-yaml``.
    """
    user_options = template_config.user_options or {}
    required = sorted(key for key, value in user_options.items() if "default" not in value)

    additional_covariates_schema: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string"},
        "default": [],
    }
    if not template_config.allow_free_additional_continuous_covariates:
        additional_covariates_schema["maxItems"] = 0
        additional_covariates_schema["description"] = (
            "This model does not allow free additional continuous covariates; leave empty."
        )

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"ModelConfiguration for {template_config.name}",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "user_option_values": {
                "type": "object",
                "additionalProperties": False,
                "properties": dict(user_options),
                "required": required,
            },
            "additional_continuous_covariates": additional_covariates_schema,
        },
    }


def build_example_configuration(template_config: ModelTemplateConfigV2) -> dict[str, Any]:
    """Build an example model_configuration_yaml dict populated with default values.

    For each ``user_options`` entry that declares a ``default``, that value is used; entries
    without a default are emitted as ``null`` so the user can fill them in.
    ``additional_continuous_covariates`` is always an empty list.
    """
    user_options = template_config.user_options or {}
    user_option_values: dict[str, Any] = {key: spec.get("default") for key, spec in user_options.items()}
    return {
        "user_option_values": user_option_values,
        "additional_continuous_covariates": [],
    }


def example_configuration_yaml(template_config: ModelTemplateConfigV2, model_name: str) -> str:
    """The example configuration of ``build_example_configuration`` as YAML, with each option explained.

    Every option gets its description, allowed values and default as comments above it, so the file
    shows how to change the model without looking up its schema.
    """
    import textwrap

    import yaml

    def comment(text: str, indent: str) -> list[str]:
        return [
            f"{indent}# {line}"
            for line in textwrap.wrap(" ".join(text.split()), width=96 - len(indent), break_on_hyphens=False)
        ]

    lines = [
        f"# Options for {model_name}, used as --model-configuration-yaml.",
        "# Change a value to change how the model runs. The values below are the model's defaults.",
        "# `chap model schema` shows every option in full.",
        "",
        "user_option_values:",
    ]
    for name, spec in (template_config.user_options or {}).items():
        lines += comment(str(spec.get("description") or spec.get("title") or ""), "  ")
        details = []
        if "enum" in spec:
            details.append("One of: " + ", ".join(str(value) for value in spec["enum"]) + ".")
        elif "type" in spec:
            details.append(f"Type: {spec['type']}.")
        bounds = [
            f"{word} {spec[key]}" for key, word in (("minimum", "at least"), ("maximum", "at most")) if key in spec
        ]
        if bounds:
            details.append(f"Must be {' and '.join(bounds)}.")
        default = spec.get("default")
        details.append("Required: fill in a value." if default is None else f"Default: {json_text(default)}.")
        lines += comment(" ".join(details), "  ")
        dumped = yaml.safe_dump({name: default}, sort_keys=False, width=1000).rstrip("\n")
        lines += [f"  {line}" for line in dumped.splitlines()]
    lines.append("")
    if template_config.allow_free_additional_continuous_covariates:
        lines.append("# Extra dataset columns to give the model as continuous covariates, such as [humidity].")
    else:
        lines.append("# This model takes no extra covariates; leave this empty.")
    lines.append("additional_continuous_covariates: []")
    return "\n".join(lines) + "\n"


def json_text(value: Any) -> str:
    """A default value as it reads in a comment: strings quoted, others as JSON."""
    import json

    return json.dumps(value)


def schema_cmd(
    model_name: ModelNameArg,
    output_file: Annotated[
        Path | None,
        Parameter(
            "--output-file",
            help="Optional path to write the YAML output to. Prints to stdout if omitted.",
        ),
    ] = None,
    example: Annotated[
        bool,
        Parameter(
            "--example",
            help="Emit an example configuration filled with default values instead of the schema.",
        ),
    ] = False,
    run_config: RunConfigArg = RunConfig(),
):
    """Print the model configuration schema for a model.

    The schema describes the structure of a ``--model-configuration-yaml`` file accepted by
    commands like ``chap evaluate``. The ``user_options`` declared by
    the model template are wrapped inside the ``user_option_values`` field, and the schema
    also includes the ``additional_continuous_covariates`` field.

    With ``--example``, prints a runnable example configuration with default values instead
    of the JSON schema.

    Examples:
        chap model schema https://github.com/dhis2-chap/minimalist_example_lag
        chap model schema ./my_model --output-file schema.yaml
        chap model schema ./my_model --example
    """
    import yaml

    from chap_core.cli_endpoints._common import inspect_model_template

    with inspect_model_template(model_name, run_config) as template:
        builder = build_example_configuration if example else build_model_configuration_schema
        payload = builder(template.model_template_config)

    rendered = yaml.safe_dump(payload, sort_keys=False)
    if output_file is None:
        sys.stdout.write(rendered)
    else:
        output_file.write_text(rendered, encoding="utf-8")
        logger.info(f"Output written to {output_file}")


def register_commands(app):
    from cyclopts import App

    model_app = App(name="model", help="Model introspection commands.")
    model_app.command(name="schema")(schema_cmd)
    app.command(model_app)

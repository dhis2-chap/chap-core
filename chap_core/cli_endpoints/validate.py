"""Validate commands for CHAP CLI."""

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
from cyclopts import Parameter

from chap_core.cli_endpoints._common import discover_geojson, load_dataset_from_csv
from chap_core.models.model_template import ModelTemplate
from chap_core.services.dataset_validation import ValidationIssue, validate_dataset

logger = logging.getLogger(__name__)


def validate_cmd(
    dataset_csv: Annotated[
        Path,
        Parameter(
            help="Path to CSV file containing disease data with columns: time_period, "
            "location, disease_cases, and climate covariates"
        ),
    ],
    model_name: Annotated[
        Optional[str],
        Parameter(help="Model path (local directory) or GitHub URL to validate dataset against"),
    ] = None,
    data_source_mapping: Annotated[
        Optional[Path],
        Parameter(
            help="Path to JSON file mapping model covariate names to CSV column names. "
            'Format: {"model_name": "csv_column"}'
        ),
    ] = None,
):
    """Validate a CSV dataset for CHAP compatibility.

    Checks dataset structure and content, reporting any issues found.
    Optionally validates that the dataset meets a specific model's requirements.

    Examples:
        # Basic validation
        chap validate --dataset-csv ./data/vietnam.csv

        # Validate against a model
        chap validate --dataset-csv ./data/vietnam.csv \\
            --model-name https://github.com/dhis2-chap/minimalist_example_r
    """
    column_mapping = None
    if data_source_mapping is not None:
        logger.info(f"Loading column mapping from {data_source_mapping}")
        with open(data_source_mapping) as f:
            column_mapping = json.load(f)

    raw_df = pd.read_csv(dataset_csv)
    if column_mapping is not None:
        raw_df.rename(columns={v: k for k, v in column_mapping.items()}, inplace=True)

    for required_col in ("time_period", "location"):
        if required_col not in raw_df.columns:
            print(f"Error: Required column '{required_col}' not found in CSV.")
            print(f"Available columns: {list(raw_df.columns)}")
            sys.exit(1)

    try:
        geojson_path = discover_geojson(dataset_csv)
        dataset = load_dataset_from_csv(dataset_csv, geojson_path, column_mapping)
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        _report_period_gaps(raw_df)
        sys.exit(1)

    model_template_config = None
    if model_name is not None:
        logger.info(f"Loading model template from {model_name}")
        template = ModelTemplate.from_directory_or_github_url(model_name)
        model_template_config = template.model_template_config

    issues = validate_dataset(dataset, raw_df=raw_df, model_template_config=model_template_config)

    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for issue in warnings:
            _print_issue(issue)

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for issue in errors:
            _print_issue(issue)
        sys.exit(1)

    if not issues:
        print("Validation passed: no issues found.")


def _print_issue(issue: ValidationIssue):
    parts = [f"  - {issue.message}"]
    if issue.location:
        parts.append(f"    Location: {issue.location}")
    if issue.time_periods:
        parts.append(f"    Time periods: {_format_period_ranges(issue.time_periods)}")
    print("\n".join(parts))


def _format_period_ranges(periods: list[str]) -> str:
    """Group consecutive periods into ranges for compact display."""
    if not periods:
        return ""
    if len(periods) == 1:
        return periods[0]

    ranges = []
    range_start = 0
    for i in range(1, len(periods)):
        if not _is_adjacent(periods[i - 1], periods[i]):
            ranges.append((range_start, i - 1))
            range_start = i
    ranges.append((range_start, len(periods) - 1))

    parts = []
    for start_idx, end_idx in ranges:
        if start_idx == end_idx:
            parts.append(periods[start_idx])
        else:
            count = end_idx - start_idx + 1
            parts.append(f"{periods[start_idx]} to {periods[end_idx]} ({count} periods)")
    return ", ".join(parts)


def _is_adjacent(a: str, b: str) -> bool:
    """Check if two period strings are adjacent (e.g. 2008-01 and 2008-02)."""
    from chap_core.time_period import TimePeriod

    try:
        pa = TimePeriod.parse(a)
        pb = TimePeriod.parse(b)
        return bool(pb == pa + pa.time_delta)
    except Exception:
        return False


def _report_period_gaps(raw_df: pd.DataFrame):
    """Analyze and report per-location period gaps when dataset loading fails."""
    from chap_core.time_period import TimePeriod

    for location in sorted(raw_df["location"].unique()):
        loc_periods = sorted(raw_df[raw_df["location"] == location]["time_period"].unique())
        if len(loc_periods) < 2:
            continue

        try:
            parsed = [TimePeriod.parse(p) for p in loc_periods]
        except Exception:
            continue

        delta = parsed[0].time_delta
        missing = []
        for p1, p2 in zip(parsed, parsed[1:]):
            expected = p1 + delta
            while expected != p2:
                missing.append(expected.to_string())
                expected = expected + delta
                if len(missing) > 1000:
                    break

        if missing:
            formatted = _format_period_ranges(missing)
            print(f"  Location '{location}': missing {formatted}")


def register_commands(app):
    """Register validate commands with the CLI app."""
    app.command(name="validate")(validate_cmd)
